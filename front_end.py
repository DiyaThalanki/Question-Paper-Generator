import streamlit as st
from app import Vectorstore,run_chatbot

def main():

    # App title
    st.title("Question Paper Generator")

    # Description
    st.write("""
    This application allows you to upload documents from which questions will be generated automatically.
    Please upload the study material or any relevant text document.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

    raw_document={"title":"","content":""}

    # Process the uploaded file
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        # Read the content of the uploaded file
        file_content = None
        try:
            if uploaded_file.type == "text/plain":
                file_content = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                file_content = "".join(page.extract_text() for page in pdf_reader.pages)
                raw_document["content"]=file_content
                raw_document["title"] = uploaded_file.name
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                from docx import Document
                doc = Document(uploaded_file)
                file_content = "\n".join(paragraph.text for paragraph in doc.paragraphs)
                raw_document["content"]=file_content
                raw_document["title"] = uploaded_file.name
        except Exception as e:
            st.error(f"Failed to process the file. Error: {str(e)}")
            raw_document = {"title": "", "content": ""} 
        
        # Show file content and additional options
        if file_content:
            st.write("File Content Preview:")
            st.text_area("Preview", file_content, height=300)

            # Generate questions button
            if st.button("Generate Questions"):
                st.info("Generating questions... Please wait.")
                questions = generate_questions(raw_document)
                st.write("Generated Questions:")
                filtered_questions = [question.message for question in questions]
                st.write(filtered_questions)
                # for idx, question in enumerate(questions, start=1):
                #     st.write(f"{idx}. {question}")

def generate_questions(raw_document):
    """
    Mock function to simulate question generation from text.
    Replace with your actual question generation logic.
    """
    chat_history = run_chatbot(raw_document,"How many memebers per team?")
    print(chat_history)
    return chat_history
    

if __name__ == "__main__":
    main()
