#!/usr/bin/env bash
# Build AuraExample.app and package it into a .dmg that contains a
# drag-to-Applications alias. Requires macOS 26 (AgentCrew) + an Xcode signing
# identity. The app is development-signed (runs on this Mac).
#
# CONFIG defaults to Debug: the Release *archive* currently trips a known
# "cannot link directly with 'SwiftUICore'" linker error; Debug builds + signs
# cleanly. Override with `CONFIG=Release ./scripts/build-app.sh` once that is fixed.
#
# For wider distribution, re-sign the .app with a Developer ID identity and
# notarize the .dmg (see the note printed at the end).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="$ROOT/Sources/AuraExample/AuraExample.xcodeproj"
SCHEME="AuraExample"
CONFIG="${CONFIG:-Debug}"
VOL="AuraExample"

BUILD="$ROOT/build"
DD="$BUILD/dd"
STAGE="$BUILD/dmg"
DMG="$BUILD/AuraExample.dmg"

rm -rf "$BUILD"; mkdir -p "$BUILD"

echo "==> Building AuraExample ($CONFIG)…"
xcodebuild -project "$PROJ" -scheme "$SCHEME" -configuration "$CONFIG" \
  -destination 'platform=macOS' -skipMacroValidation \
  -derivedDataPath "$DD" build

APP="$DD/Build/Products/$CONFIG/AuraExample.app"
[ -d "$APP" ] || { echo "error: app not found at $APP"; exit 1; }

echo "==> Staging .dmg contents (app + /Applications alias)…"
rm -rf "$STAGE"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"   # drag-to-install alias

echo "==> Creating .dmg…"
rm -f "$DMG"
hdiutil create -volname "$VOL" -srcfolder "$STAGE" -ov -format UDZO "$DMG"

echo "==> Done: $DMG"
echo "Note: $CONFIG build, development-signed — runs on this Mac. To distribute to"
echo "      others, build a Developer-ID-signed .app and notarize the .dmg:"
echo "        xcrun notarytool submit \"$DMG\" --keychain-profile <profile> --wait"
echo "        xcrun stapler staple \"$DMG\""
