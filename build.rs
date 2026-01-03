fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let has_gpt_sovits = std::env::var("CARGO_FEATURE_GPT_SOVITS").is_ok();

    if target_os == "windows" && target_env == "msvc" && has_gpt_sovits {
        // Force the final binary to keep a dependency on the CUDA backend DLL.
        println!("cargo:rustc-link-arg=/INCLUDE:?warp_size@cuda@at@@YAHXZ");
    }

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GPT_SOVITS");
}
