from pathlib import Path

from pypdf import PdfReader


MATERIALS_DIR = Path(r"D:\model\paper_templates\CTSF")
OUT_DIR = Path(r"D:\model\CTSF\paper_rewriting_output\reference_materials\pdf_text")
SUMMARY_PATH = Path(
    r"D:\model\CTSF\paper_rewriting_output\reference_materials\pdf_extract_summary.md"
)


def compact(text: str, limit: int = 900) -> str:
    return " ".join(text.split())[:limit].replace("|", "/")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# PDF Text Extraction Summary", ""]
    count = 0
    for pdf in sorted(MATERIALS_DIR.glob("*.pdf")):
        count += 1
        try:
            reader = PdfReader(str(pdf))
            pages = []
            for idx, page in enumerate(reader.pages, start=1):
                try:
                    pages.append(page.extract_text() or "")
                except Exception as exc:
                    pages.append(f"[extract error on page {idx}: {exc}]")
            full_text = "\n\n".join(pages)
            (OUT_DIR / f"{pdf.stem}.txt").write_text(full_text, encoding="utf-8")
            lines.extend(
                [
                    f"## {pdf.name}",
                    f"- Pages: {len(reader.pages)}",
                    f"- Characters extracted: {len(full_text)}",
                    f"- Opening text: {compact(full_text)}",
                    "",
                ]
            )
        except Exception as exc:
            lines.extend([f"## {pdf.name}", f"- Extraction failed: {exc}", ""])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"extracted {count} pdfs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
