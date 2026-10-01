"""Generate eval/sample_docs/employee_handbook.pdf, a multi-page PDF for testing page citations.

Run: python eval/make_sample_pdf.py
"""

from pathlib import Path

import pymupdf

PAGES = [
    """Halcyon Outdoor Gear Employee Handbook

Welcome to Halcyon. This handbook covers working hours, leave, and benefits for all
employees at our Reno headquarters and our four retail stores.

Halcyon was founded in 2011 by Maya Okafor and Dev Lindqvist in a garage in Truckee,
California. The company moved its headquarters to Reno, Nevada in 2017.""",
    """Working Hours and Remote Work

Core hours at headquarters are 10:00 to 15:00 Pacific Time. Outside core hours,
employees may arrange their schedules with their manager.

Headquarters staff may work remotely up to two days per week. Retail store staff
work scheduled shifts and are not eligible for remote work.""",
    """Paid Time Off

Full-time employees accrue 1.75 days of paid time off per month, which is 21 days per
year. Up to 5 unused days can be carried over into the next calendar year.

Every employee also receives two paid "Trail Days" per year to spend outdoors.
Trail Days do not carry over.""",
    """Benefits

Employees receive a 40% discount on Halcyon-branded gear and a 20% discount on
third-party brands sold in our stores.

The company matches 401(k) contributions dollar for dollar up to 4% of salary.
New employees become eligible for the 401(k) match after 90 days of employment.""",
]


def main() -> None:
    out = Path(__file__).parent / "sample_docs" / "employee_handbook.pdf"
    doc = pymupdf.open()
    for text in PAGES:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 770), text, fontsize=11)
    doc.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
