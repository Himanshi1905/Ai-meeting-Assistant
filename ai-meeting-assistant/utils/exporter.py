import os
from fpdf import FPDF

class MeetingReportPDF(FPDF):
    """
    Custom subclass of FPDF to handle layout headers, footers, and page numbers.
    """
    def header(self):
        # Render a clean header on each page
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, 'AI Meeting Assistant - Summary & Action Items Report', 0, 1, 'R')
        self.set_draw_color(220, 220, 220)
        self.line(10, 18, 200, 18)
        self.ln(4)

    def footer(self):
        # Render a page numbering footer
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_pdf_report(meeting: dict) -> bytes:
    """
    Generates a professionally structured PDF from a meeting record.
    Returns the document as raw bytes.
    """
    pdf = MeetingReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # 1. Document Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(255, 75, 75)  # Matches Streamlit crimson theme color
    title_text = meeting.get("title", "Meeting Report")
    pdf.multi_cell(0, 10, title_text)
    pdf.ln(4)
    
    # 2. Metadata Block
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(100, 100, 100)
    
    duration_sec = meeting.get("duration", 0.0)
    duration_mins = int(duration_sec // 60)
    duration_secs = int(duration_sec % 60)
    duration_str = f"{duration_mins:02d}:{duration_secs:02d}"
    
    metadata_text = (
        f"Source Input: {meeting.get('source', 'N/A')}\n"
        f"Transcription Backend: {meeting.get('backend', 'N/A').upper()}\n"
        f"Audio Duration: {duration_str}\n"
        f"Processed Date: {meeting.get('created_at', 'N/A')}"
    )
    pdf.multi_cell(0, 5, metadata_text)
    pdf.ln(6)
    
    # Draw separator line
    pdf.set_draw_color(240, 240, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    # 3. Meeting Summary & Overview
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 10, '1. Meeting Overview & Summary', 0, 1)
    pdf.ln(1)
    
    pdf.set_font('Helvetica', '', 10.5)
    pdf.set_text_color(60, 60, 60)
    summary_text = meeting.get("summary_markdown", "No summary available.")
    
    # Strip basic markdown headers and bullet markers for simple rendering
    clean_summary = summary_text.replace('#', '').replace('**', '').replace('__', '').replace('* ', '- ').replace('`', '')
    pdf.multi_cell(0, 5.5, clean_summary)
    pdf.ln(6)
    
    # 4. Action Items list
    analysis = meeting.get("analysis", {})
    action_items = analysis.get("action_items", [])
    decisions = analysis.get("decisions", [])
    open_questions = analysis.get("open_questions", [])
    
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 10, '2. Extracted Action Items', 0, 1)
    pdf.ln(1)
    
    if not action_items:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, '   No action items extracted from the transcript.', 0, 1)
        pdf.ln(4)
    else:
        for i, item in enumerate(action_items, 1):
            pdf.set_font('Helvetica', 'B', 10.5)
            pdf.set_text_color(60, 60, 60)
            desc = item.get("description", "")
            
            # Print task description
            pdf.multi_cell(0, 5.5, f"   [ ]  {desc}")
            
            # Print owner/deadline detail sub-block
            owner = item.get("owner")
            deadline = item.get("deadline")
            details = []
            if owner:
                details.append(f"Owner: {owner}")
            if deadline:
                details.append(f"Deadline: {deadline}")
                
            if details:
                pdf.set_font('Helvetica', 'I', 9.5)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(0, 5, f"        " + "  |  ".join(details), 0, 1)
                pdf.ln(1)
        pdf.ln(4)
        
    # 5. Decisions list
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 10, '3. Decisions Reached', 0, 1)
    pdf.ln(1)
    
    if not decisions:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, '   No key decisions recorded.', 0, 1)
        pdf.ln(4)
    else:
        for i, dec in enumerate(decisions, 1):
            pdf.set_font('Helvetica', '', 10.5)
            pdf.set_text_color(60, 60, 60)
            desc = dec.get("description", "")
            context = dec.get("context", "")
            
            pdf.multi_cell(0, 5.5, f"   -  {desc}")
            if context:
                pdf.set_font('Helvetica', 'I', 9.5)
                pdf.set_text_color(120, 120, 120)
                pdf.multi_cell(0, 5, f"      Context: {context}")
                pdf.ln(1)
        pdf.ln(4)
        
    # 6. Open Questions
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 10, '4. Open Questions & Blockers', 0, 1)
    pdf.ln(1)
    
    if not open_questions:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, '   No unresolved questions or blockers.', 0, 1)
        pdf.ln(4)
    else:
        for i, q in enumerate(open_questions, 1):
            pdf.set_font('Helvetica', '', 10.5)
            pdf.set_text_color(60, 60, 60)
            question = q.get("question", "")
            context = q.get("context", "")
            
            pdf.multi_cell(0, 5.5, f"   ?  {question}")
            if context:
                pdf.set_font('Helvetica', 'I', 9.5)
                pdf.set_text_color(120, 120, 120)
                pdf.multi_cell(0, 5, f"      Context/Details: {context}")
                pdf.ln(1)
        pdf.ln(4)
        
    return pdf.output()
