"""Generate the synthetic, internally consistent demo corpus.  python scripts/make_synthetic_docs.py
Every page carries a SYNTHETIC banner + page footer (also exercises the header/footer cleaner)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from config import DOCS_DIR

INST = "Northfield Institute of Technology (NIT) - SYNTHETIC DEMO DATA, not real college policy"
ss = getSampleStyleSheet()

DOCS = {
 "regulations__ALL__0.pdf": [
  ("Academic Regulations 2026-27", [
   "Attendance: Every student must maintain a minimum of 75% attendance in each course to be eligible for the end-semester examination. Students with attendance between 65% and 74% may apply for condonation with a medical certificate and a fee of Rs. 500. Attendance below 65% results in detention in that course.",
   "Grading: The institute uses a 10-point scale. O = 10, A+ = 9, A = 8, B+ = 7, B = 6, C = 5, F = 0 (fail). SGPA is the credit-weighted average of grade points in a semester. CGPA is the credit-weighted average over all completed semesters.",
   "Minimum passing: A student must score at least 40% in the end-semester examination and 40% overall to pass a course."]),
  ("Backlogs and Promotion", [
   "A student may carry a maximum of 4 active backlogs into the next academic year. Backlog exams are conducted twice a year, in January and June.",
   "Leave: Students may apply for a maximum of 10 days of leave per semester through the department office, with the faculty advisor's approval."])],
 "calendar__ALL__0__ALL__2026-27.pdf": [
  ("Academic Calendar - Odd Semester 2026-27", [
   "Semester begins: 3 August 2026. Mid-semester examinations: 21 September to 26 September 2026. Last instructional day: 20 November 2026.",
   "End-semester examinations: 2 December to 18 December 2026. Results are announced on 8 January 2027. Winter break: 21 December 2026 to 1 January 2027."])],
 "curriculum__CSE__5.pdf": [
  ("B.Tech CSE - Semester 5 Curriculum", [
   "Subjects in Semester 5 for Computer Science and Engineering (CSE): Machine Learning (4 credits), Database Management Systems (4 credits), Computer Networks (3 credits), Operating Systems (3 credits), Design and Analysis of Algorithms (4 credits), and Professional Ethics (2 credits). Total credits: 20.",
   "Labs in Semester 5: Machine Learning Lab (1 credit) and DBMS Lab (1 credit)."])],
 "curriculum__ECE__5.pdf": [
  ("B.Tech ECE - Semester 5 Curriculum", [
   "Subjects in Semester 5 for Electronics and Communication Engineering (ECE): Digital Signal Processing (4 credits), VLSI Design (4 credits), Microwave Engineering (3 credits), Control Systems (3 credits), Embedded Systems (3 credits). Total credits: 17."])],
 "handbook__CSE__5__Machine-Learning.pdf": [
  ("Subject Handbook - Machine Learning (CSE, Semester 5)", [
   "Course outline: Supervised learning, decision trees, support vector machines, neural networks, clustering and model evaluation. Evaluation: Mid-semester 20%, assignments 20%, end-semester 60%. Prerequisite: Linear Algebra and Probability."])],
 "examination__ALL__0.pdf": [
  ("Examination Guidelines", [
   "Students must carry the institute ID card to every examination. Mobile phones and smart watches are not permitted in the hall. Reporting time is 30 minutes before the exam. Students arriving after 30 minutes from the start are not allowed to enter.",
   "Re-evaluation: Students may apply for re-evaluation within 7 days of result publication, at Rs. 300 per course."])],
 "internship__ALL__0.pdf": [
  ("Internship Guidelines", [
   "Internships are mandatory for B.Tech students after Semester 6, with a minimum duration of 6 weeks. Students must submit an internship report and a completion certificate within 2 weeks of returning. The internship carries 2 credits."])],
 "placement__ALL__0.pdf": [
  ("Placement Guidelines", [
   "Eligibility for campus placements: minimum CGPA of 6.5, no active backlogs, and attendance of at least 75%. Students placed in a Dream company (package above Rs. 12 LPA) may not sit for further drives. The placement season begins in August of the final year."])],
 "scholarship__ALL__0.pdf": [
  ("Scholarship Guidelines", [
   "Merit Scholarship: Students with a CGPA of 9.0 or above receive a 50% tuition waiver for the following year. Students with CGPA between 8.0 and 8.99 receive a 25% waiver. Applications close on 15 September each year at the accounts office.",
   "Need-based aid requires an income certificate and a letter from the faculty advisor."])],
 "faq__ALL__0.pdf": [
  ("Student FAQ", [
   "Q: Where can I get my bonafide certificate? A: Apply at the academic office; it is issued within 3 working days.",
   "Q: How do I replace a lost ID card? A: Submit an application and pay Rs. 200 at the administrative office.",
   "Q: What are the library hours? A: The library is open from 8:30 AM to 8:00 PM on weekdays and 9:00 AM to 4:00 PM on Saturdays."])],
}


def header_footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.drawString(40, 815, INST)
    canvas.drawString(40, 25, f"NIT Academic Office - Page {doc.page}")


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, sections in DOCS.items():
        doc = SimpleDocTemplate(str(DOCS_DIR / fname), pagesize=A4)
        story = []
        for i, (title, paras) in enumerate(sections):
            if i:
                story.append(PageBreak())      # one section per page -> distinct page citations
            story.append(Paragraph(title, ss["Heading1"]))
            for p in paras:
                story += [Paragraph(p, ss["BodyText"]), Spacer(1, 10)]
        doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
        print("wrote", fname)


if __name__ == "__main__":
    main()
