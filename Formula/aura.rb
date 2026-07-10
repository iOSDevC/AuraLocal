# Homebrew formula for the `aura` CLI (personal tap). Copy this into a
# `homebrew-aura` tap repo as Formula/aura.rb and publish. See docs/HOMEBREW.md.
#
# Installs a prebuilt tarball (aura binary + llama.framework, co-located) produced
# by scripts/package-cli.sh — no SwiftPM/network is needed at install time.
class Aura < Formula
  desc "AuraLocal integration CLI — hybrid on-device + remote LLM features"
  homepage "https://github.com/iOSDevC/AuraLocal"
  version "0.1.0"
  url "https://github.com/iOSDevC/AuraLocal/releases/download/v0.1.0/aura-v0.1.0-macos-arm64.tar.gz"
  # sha256 of the tarball printed by `./scripts/package-cli.sh 0.1.0`.
  # Recompute (shasum -a 256) if you rebuild the tarball, then update this.
  sha256 "e50339070babd56b0b934d4353b6f37e994c350d25a64a96468090444e8eb5de"
  license "MIT"

  depends_on :macos
  depends_on arch: :arm64

  def install
    # `aura` links @rpath/llama.framework and carries an @loader_path rpath, so the
    # framework must sit next to the real binary. Keep both in libexec; link into bin.
    libexec.install "aura", "llama.framework"
    bin.install_symlink libexec/"aura"
  end

  test do
    assert_match "system.vision.ocr", shell_output("#{bin}/aura tools")
  end
end
