import argparse
import time
from playwright.sync_api import sync_playwright

def wait_for_game_load(page, max_wait_sec=60):
    print("Waiting for game download and extraction to finish...")
    start = time.time()
    last_status = ""

    while (time.time() - start) < max_wait_sec:
        status = page.evaluate("""() => {
            const dlBar = document.getElementById('download-bar');
            const exBar = document.getElementById('extract-bar');
            const dlText = document.getElementById('download-text');
            const exText = document.getElementById('extract-text');

            const isVisible = (el) => el && el.offsetParent !== null && getComputedStyle(el).display !== 'none';

            if (isVisible(dlBar) || isVisible(exBar)) {
                const dl = dlText ? dlText.innerText.trim() : '';
                const ex = exText ? exText.innerText.trim() : '';
                return `Downloading: ${dl || '...'} | Extracting: ${ex || '...'}`;
            }
            return 'DONE';
        }""")

        if status == "DONE":
            print("   -> Loading & extraction complete!")
            break

        if status != last_status:
            print(f"   -> {status}")
            last_status = status

        page.wait_for_timeout(500)

    if (time.time() - start) >= max_wait_sec:
        print("   -> Warning: Timed out waiting for loading bars to hide.")

    # Settle buffer for canvas title screen & audio initialization
    page.wait_for_timeout(6500)

def inspect_html_game(url, click_coords=None):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})

        print(f"Loading {url}...")
        page.goto(url)

        page.add_style_tag(
            content="#wrongRotation { display: none !important; } #game { display: block !important; }"
        )

        wait_for_game_load(page)

        if click_coords:
            x, y = click_coords
            print(f"Clicking ({x}, {y}) and waiting 5 seconds...")
            page.mouse.click(x, y)
            page.wait_for_timeout(5000)
            print("Clicking center of screen (640, 360) and waiting 5 seconds...")
            page.mouse.click(640, 360)
            page.wait_for_timeout(5000)

        screenshot_path = "inspector_preview.png"
        page.screenshot(path=screenshot_path)

        report = page.evaluate("""() => {
            const canvases = Array.from(document.querySelectorAll('canvas')).map((c, i) => {
                const rect = c.getBoundingClientRect();
                return {
                    index: i,
                    id: c.id || '(no id)',
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    x: Math.round(rect.x),
                    y: Math.round(rect.y)
                };
            });

            const domElements = Array.from(document.querySelectorAll('button, a, div, span, canvas, [id]'))
                .filter(el => el.id && el.offsetParent !== null)
                .map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        tag: el.tagName.toLowerCase(),
                        id: el.id,
                        centerX: Math.round(rect.x + rect.width / 2),
                        centerY: Math.round(rect.y + rect.height / 2),
                        size: `${Math.round(rect.width)}x${Math.round(rect.height)}`
                    };
                });

            const engines = [];
            if (window.Phaser) engines.push('Phaser');
            if (window.PIXI) engines.push('PixiJS');
            if (document.getElementById('MMFCanvas') || window.Runtime) engines.push('Clickteam Fusion (MMF2.5)');
            if (window.UnityLoader || window.unityInstance || document.getElementById('unity-canvas')) engines.push('Unity WebGL');
            if (window.cr_getC2Runtime || window.c3_runtimeInterface) engines.push('Construct 2/3');

            const mmfInfo = {};
            const mmfCanvas = document.getElementById('MMFCanvas');
            if (mmfCanvas) {
                for (const key of Object.keys(window)) {
                    try {
                        const val = window[key];
                        if (val && typeof val === 'object') {
                            if ('currentFrame' in val && 'nFrames' in val) {
                                mmfInfo.appVar = key;
                                mmfInfo.totalFrames = val.nFrames;
                                mmfInfo.currentFrameIndex = val.currentFrame?.index ?? 'unknown';
                                mmfInfo.frameSize = `${val.gaCxWin || '?'}x${val.gaCyWin || '?'}`;
                                mmfInfo.score = val.score;
                                mmfInfo.lives = val.lives;
                            }
                            if (val.rhObjectList && Array.isArray(val.rhObjectList)) {
                                mmfInfo.activeObjectsCount = val.rhObjectList.filter(Boolean).length;
                            }
                        }
                    } catch (e) {}
                }
            }

            const ignoredPrefixes = ['on', 'webkit', 'google', 'ga', 'ima', 'closure', 'module$', 'SDK', 'adx', 'ggeac', 'Goog'];
            const standardKeys = new Set([
                'window', 'self', 'document', 'name', 'location', 'customElements', 'history',
                'navigation', 'locationbar', 'menubar', 'personalbar', 'scrollbars', 'statusbar',
                'toolbar', 'status', 'closed', 'frames', 'length', 'top', 'opener', 'parent',
                'frameElement', 'navigator', 'origin', 'external', 'screen', 'innerWidth',
                'innerHeight', 'scrollX', 'pageXOffset', 'scrollY', 'pageYOffset', 'visualViewport',
                'screenX', 'screenY', 'outerWidth', 'outerHeight', 'devicePixelRatio', 'clientInformation',
                'screenLeft', 'screenTop', 'styleMedia', 'isSecureContext', 'crossOriginIsolated',
                'scheduler', 'performance', 'trustedTypes', 'crypto', 'indexedDB', 'localStorage',
                'sessionStorage', 'crashReport', 'cookieStore', 'caches', 'documentPictureInPicture',
                'originAgentCluster', 'viewport', 'credentialless', 'fence', 'launchQueue',
                'speechSynthesis', 'dataLayer', 'event', '1', '0'
            ]);

            const numbersAndBooleans = {};
            const arrays = {};
            const gameObjects = {};

            for (const key of Object.keys(window)) {
                if (standardKeys.has(key)) continue;
                if (ignoredPrefixes.some(prefix => key.startsWith(prefix))) continue;

                try {
                    const val = window[key];
                    if (val === null || val === undefined || typeof val === 'function') continue;

                    if (typeof val === 'number' || typeof val === 'boolean' || typeof val === 'string') {
                        numbersAndBooleans[key] = val;
                    } else if (Array.isArray(val)) {
                        const firstItem = val;
                        let sampleInfo = typeof firstItem;
                        if (firstItem && typeof firstItem === 'object') {
                            sampleInfo = firstItem.frameName || firstItem.key || Object.keys(firstItem).slice(0, 4).join(', ');
                        }
                        arrays[key] = `Length: ${val.length} (Sample: ${sampleInfo})`;
                    } else if (typeof val === 'object' && 'x' in val && 'y' in val) {
                        gameObjects[key] = {
                            x: Math.round(val.x),
                            y: Math.round(val.y),
                            visible: val.visible ?? true,
                            frame: val.frameName || val.frame || 'none'
                        };
                    }
                } catch (e) {}
            }

            return {
                title: document.title || '(No Title)',
                engines,
                canvases,
                domElements,
                mmfInfo,
                numbersAndBooleans,
                arrays,
                gameObjects
            };
        }""")

        print("\n" + "=" * 65)
        print(f"UNIVERSAL HTML GAME REPORT: {report['title']}")
        print("=" * 65)

        print(f"\n DETECTED GAME ENGINE(S):")
        print("   • " + (", ".join(report["engines"]) if report["engines"] else "Vanilla HTML5 / Custom Engine"))

        print(f"\n CANVAS ELEMENTS ({len(report['canvases'])} found):")
        for c in report["canvases"]:
            print(f"   • Canvas #{c['index']} (id='{c['id']}'): {c['width']}x{c['height']} px at ({c['x']}, {c['y']})")

        print(f"\n VISIBLE HTML ELEMENTS WITH IDs:")
        if report["domElements"]:
            for el in report["domElements"]:
                print(f"   • <{el['tag']} id='{el['id']}'> -> Center: ({el['centerX']}, {el['centerY']}) | Size: {el['size']}")
        else:
            print("   • None found")

        if report["mmfInfo"]:
            print(f"\n CLICKTEAM FUSION (MMF2.5) RUNTIME INTERNALS:")
            for k, v in report["mmfInfo"].items():
                print(f"   • {k}: {v}")

        print(f"\n EXPOSED GAME VARIABLES (Numbers, Booleans, Strings):")
        if report["numbersAndBooleans"]:
            for k, v in report["numbersAndBooleans"].items():
                print(f"   • {k}: {repr(v)}")
        else:
            print("   • None found (Engine encapsulates variables inside closures)")

        print(f"\n EXPOSED GAME ARRAYS (Grids, Tile Lists, Entities):")
        if report["arrays"]:
            for k, desc in report["arrays"].items():
                print(f"   • {k} -> {desc}")
        else:
            print("   • None found")

        print(f"\n IN-CANVAS SPRITES & BUTTONS (Phaser/PixiJS):")
        if report["gameObjects"]:
            for k, obj in report["gameObjects"].items():
                status = "VISIBLE" if obj["visible"] else "hidden"
                print(f"   • {k:25s} | Pos: ({obj['x']:4d}, {obj['y']:4d}) | Frame: {str(obj['frame']):15s} | [{status}]")
        else:
            print("   • None found (Use pixel/CNN observations or inspect canvas coordinates)")
            print(f"\n VISUAL VERIFICATION:")
            print(f"   • Screenshot saved to: {screenshot_path}")
            print("=" * 65 + "\n")
            browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect any HTML5 game for RL environment building.")
    parser.add_argument("url", nargs="?", default="http://localhost:8000/FNAF.html", help="URL of the HTML game")
    parser.add_argument("--click", nargs=2, type=int, metavar=("X", "Y"), help="Optional X Y coordinates to click after load")
    args = parser.parse_args()
    inspect_html_game(args.url, click_coords=args.click)
