import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imfusion_io import to_shared_image
from vessel_phantom import VesselSpec, generate_phantom


def main() -> int:
    try:
        import imfusion
    except ImportError:
        print("Cannot load ImFusion. Check that the SDK and its license are available.", file=sys.stderr)
        return 1

    try:
        imfusion.init()
        if not imfusion.has_gl_context():
            print("The ImFusion viewer requires an available OpenGL context.", file=sys.stderr)
            return 1

        spec = VesselSpec()
        image = to_shared_image(generate_phantom(spec), (spec.spacing_mm,) * 3)
        images = imfusion.SharedImageSet()
        images.add(image)
        imfusion.show(images, title="Vessel Stenosis Phantom", block=True)
    except imfusion.MissingLicenseError:
        print("The ImFusion viewer requires an available license.", file=sys.stderr)
        return 1
    except RuntimeError:
        print("The ImFusion viewer could not open. Check OpenGL support.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
