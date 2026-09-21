//! The Android platform wiring that a Kotlin host cannot do for a Rust
//! library, and that nothing else in an ordinary app would do either.
//!
//! Two things live here. Logging, in the `logcat` submodule: a `tracing`
//! subscriber and panic hook writing to logcat, started by
//! [`init_logging`]. And the `ndk_context` publish below, which hands the
//! platform's `JavaVM` and `Context` to the dependencies that expect to find
//! them.
//!
//! # The `ndk_context` publish
//!
//! Android gives apps no readable `/etc/resolv.conf`, so a Rust DNS stack has
//! to ask the framework for the active network's resolvers over JNI:
//! `Context.getSystemService("connectivity")`, then `getActiveNetwork`,
//! `getLinkProperties` and `getDnsServers`. `hickory-resolver` does exactly
//! that, and it finds the `JavaVM` and `Context` it needs to make those calls
//! in [`ndk_context`] — a process-global slot holding the two pointers, so a
//! crate buried in a dependency tree can reach the JVM without every API
//! between here and there passing it down.
//!
//! That slot is normally filled by `ndk-glue` or `android-activity`, which is
//! to say: by the crates used when *Rust owns the app*, as in a
//! `NativeActivity`. This SDK is the opposite arrangement — Kotlin owns the
//! Activity and loads a `cdylib` through `System.loadLibrary` — so nothing
//! ever filled it, and [`ndk_context::android_context`] is an `expect` on an
//! empty `Option`. Reaching it was fatal rather than merely broken, because
//! the release profile sets `panic = "abort"`: the panic took the whole host
//! app down with a `SIGABRT` and an `android context was not initialized`
//! abort message, from a `DefaultDispatch` worker thread, with no Rust error
//! ever reaching the caller.
//!
//! The path that got there in practice was a Lightning receive:
//! `select_available_gateway` probes *every* gateway a federation announces,
//! in parallel, to find out which are online. A federation running a gateway
//! that registers itself twice — once under `http`, once under `iroh`, which
//! is what devimint does — puts an `iroh://` URL in that list, dialing it
//! builds an iroh endpoint, and an iroh endpoint builds a DNS resolver. The
//! reachable `http` gateway would have won the subsequent ranking; the
//! process never survived the survey to get there.
//!
//! Guardians are the same story through a different door: `connect_guardian`
//! and `connect_gateway` share one scheme-to-connector map, so an `iroh://`
//! URL in an invite code would abort at join, before any gateway exists.
//! Filling the slot once, here, covers both — and every future dependency
//! that wants the platform's DNS configuration — rather than teaching one
//! call site to avoid one transport.
//!
//! # Why `JNI_OnLoad`, and its limit
//!
//! `JNI_OnLoad` is the one hook that is handed the `JavaVM`: ART's library
//! loader looks it up and calls it when `System.loadLibrary` maps this
//! library. There is no way to express "pass me an
//! `android.content.Context`" through UniFFI's type system, so this is the
//! only entry point that reaches the VM at all.
//!
//! **uniffi's bindings alone would never call it.** UniFFI's Kotlin loads this
//! library through JNA's `Native.register`, and JNA tries a plain `dlopen`
//! first, falling back to `System.loadLibrary` only if that fails. For a
//! library shipped in the APK's `jniLibs` the `dlopen` succeeds, so ART's
//! loader is never involved and `JNI_OnLoad` is never called. That was
//! confirmed on a device: the library loaded and served SDK calls, `nm -D`
//! showed `JNI_OnLoad` exported, and logcat showed ART's loader handling only
//! JNA's own `libjnidispatch.so`.
//!
//! So `scripts/generate-kotlin-bindings.sh` patches the generated loader to
//! call `System.loadLibrary` ahead of each `Native.register`. ART then loads
//! the library and calls this, and JNA's `dlopen` that follows finds the same
//! copy already loaded — also confirmed on a device: one copy, and the
//! publish below succeeds. Every app embedding the SDK gets this without
//! doing anything. The script fails, rather than skipping the patch, if a
//! uniffi upgrade changes the lines it anchors on, because without it this
//! function silently never runs.
//!
//! It gives us the VM but not a `Context`, so the `Context` is fetched from
//! Rust: `ActivityThread.currentApplication()` returns this process's own
//! `Application`, with `AppGlobals.getInitialApplication()` as a fallback for
//! hosts where the former is not yet populated.

// The crate's one exception to `deny(unsafe_code)` (see `lib.rs`), covering
// this module and `logcat` below it. Everything unsafe in either is FFI:
// declaring C functions from liblog and libc, trusting the `JavaVM` pointer
// the runtime hands `JNI_OnLoad`, and publishing that pointer with the
// application `Context` into `ndk_context`'s global slot. None of it has a
// safe formulation — `ndk-context` exists precisely to hold raw pointers — and
// all of it is confined to these two files.
#![allow(unsafe_code)]

mod logcat;

use std::ffi::c_void;
use std::sync::Once;

use jni::{JavaVM, jni_sig, jni_str};
use tracing::{info, warn};

/// Starts routing logs to logcat. See the `logcat` module for what that
/// installs and how to change its filter on a device.
///
/// Called from [`JNI_OnLoad`], the earliest point there is, and again at the
/// start of `create_fedimint_sdk`, the first call a host is certain to make.
/// Idempotent, so both may run, in either order.
///
/// Deliberately not dependent on `JNI_OnLoad` alone. The generated bindings
/// load the library through ART so that it runs, but that load is best
/// effort: if it fails, JNA's own `dlopen` loads the library instead and
/// `JNI_OnLoad` never runs — the one situation in which logs matter most.
/// Logging needs nothing from the JVM, only `liblog`, so unlike the
/// `ndk_context` publish below it can still start from an ordinary SDK call.
pub(crate) fn init_logging() {
    logcat::init();
}

/// Runs the publish exactly once.
///
/// [`ndk_context::initialize_android_context`] asserts that the slot was
/// empty, so a second call aborts the process — which a repeated
/// `JNI_OnLoad` (several `System.loadLibrary` calls, or a host that already
/// initialized the slot itself) would otherwise cause. Guarding the whole
/// body rather than checking the slot keeps that decision in one place, and
/// costs nothing: this runs once per process at library load.
static PUBLISHED: Once = Once::new();

/// Called by the Android runtime when this library is loaded.
///
/// Returns the JNI version this library needs of its host. The return value
/// is not decoration: a version the runtime does not recognise makes
/// `System.loadLibrary` fail outright, so it is the plain 1.6 constant rather
/// than anything derived.
///
/// **This function must not panic.** `panic = "abort"` is on in release, so a
/// panic here would abort during `System.loadLibrary` — turning a crash that
/// today needs an `iroh://` URL to trigger into one that happens on every
/// launch, on every code path that touches the SDK. Every step below is
/// therefore allowed to fail, and failure leaves the slot empty, which is
/// exactly the behaviour that shipped before this module existed.
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub extern "system" fn JNI_OnLoad(
    vm: *mut jni::sys::JavaVM,
    _reserved: *mut c_void,
) -> jni::sys::jint {
    // First, so that everything below — and everything the host does next —
    // is logged. When this runs it is the earliest point logging can start.
    init_logging();

    PUBLISHED.call_once(|| {
        // Logged before the attempt, not only after it. Whether this function
        // runs at all is a property of how the host loaded the library —
        // `System.loadLibrary` calls `JNI_OnLoad`, a bare `dlopen` does not,
        // and JNA reaches for `dlopen` first — so "did we get here" is worth
        // being able to answer separately from "did it work".
        info!("JNI_OnLoad: publishing the Android context");
        match publish(vm) {
            Ok(()) => info!("JNI_OnLoad: published the Android JavaVM and Context for ndk-context"),
            // Deliberately not fatal: see the note on panicking above. A miss
            // leaves anything that reads the platform's DNS configuration
            // aborting as it did before, which is bad but is not made better
            // by also refusing to load the library.
            Err(err) => warn!(
                %err,
                "JNI_OnLoad: could not publish the Android Context; \
                 code that reads the platform DNS configuration will abort"
            ),
        }
    });
    jni::sys::JNI_VERSION_1_6
}

/// Fetches this process's `Application` and hands it, with the `JavaVM`, to
/// [`ndk_context`].
fn publish(vm: *mut jni::sys::JavaVM) -> Result<(), jni::errors::Error> {
    // SAFETY: the runtime calls `JNI_OnLoad` with a valid `JavaVM` pointer,
    // and that VM outlives every thread in the process — including whichever
    // thread later reads it back out of `ndk_context`. `JavaVM` here is a
    // plain wrapper around the pointer with no `Drop`, so this borrows rather
    // than takes ownership of anything.
    let handle = unsafe { JavaVM::from_raw(vm) };

    // The calling thread is a Java thread already, so this reuses its
    // existing attachment rather than making one. Exceptions thrown inside
    // the closure are caught by `attach_current_thread` and returned as
    // `Error::JavaException`, so no exception is left pending on the host's
    // thread when a lookup fails.
    let context = handle.attach_current_thread(|env| {
        // `ActivityThread.currentApplication()` is the documented-by-usage
        // way to reach the process's own `Application` without being handed
        // one. It is hidden rather than public API, and has been present and
        // stable since the framework's earliest versions; the alternative is
        // requiring the host to call an initializer, which is the coupling
        // this module exists to avoid.
        let class = env.find_class(jni_str!("android/app/ActivityThread"))?;
        let mut app = env
            .call_static_method(
                &class,
                jni_str!("currentApplication"),
                jni_sig!("()Landroid/app/Application;"),
                &[],
            )?
            .l()?;

        // Null when this library is loaded before the `Application` is in
        // place — from a `ContentProvider`, say, or during
        // `attachBaseContext`. `AppGlobals` reads the same field through a
        // different accessor and is populated marginally earlier.
        if app.is_null() {
            let class = env.find_class(jni_str!("android/app/AppGlobals"))?;
            app = env
                .call_static_method(
                    &class,
                    jni_str!("getInitialApplication"),
                    jni_sig!("()Landroid/app/Application;"),
                    &[],
                )?
                .l()?;
        }

        if app.is_null() {
            return Err(jni::errors::Error::NullPtr(
                "no Application is in place for this process yet",
            ));
        }

        // A global reference, not the local one: `ndk_context` stores a raw
        // pointer that is dereferenced much later and from other threads,
        // and a local reference dies when this JNI frame returns.
        env.new_global_ref(&app)
    })?;

    // SAFETY: `PUBLISHED` makes this the only call in the process, which is
    // what `initialize_android_context` requires — it asserts the slot was
    // empty and would abort otherwise.
    //
    // `into_raw` deliberately leaks the global reference. `ndk_context` hands
    // out copies of the pointer for the rest of the process's life and has no
    // way to signal that the last reader is done, so there is no point at
    // which deleting it would be safe. It is the *application* context, which
    // lives as long as the process regardless, so the leak costs one
    // reference rather than keeping an Activity alive.
    unsafe {
        ndk_context::initialize_android_context(vm.cast(), context.into_raw().cast());
    }
    Ok(())
}
