from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os

class PdfConverter:
    def __init__(self, file_path):
       self.file_path = file_path

    # convert pdf file to a string which has space among words
    def convert_pdf_to_txt(self):
       rsrcmgr = PDFResourceManager()
       retstr = StringIO()
       # codec = 'utf-8'  # 'utf16','utf-8'
       laparams = LAParams()
       # device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
       device = TextConverter(rsrcmgr, retstr, laparams=laparams)
       fp = open(self.file_path, 'rb')
       interpreter = PDFPageInterpreter(rsrcmgr, device)
       password = ""
       maxpages = 0
       caching = True
       pagenos = set()
       for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
           interpreter.process_page(page)
       fp.close()
       device.close()
       str = retstr.getvalue()
       retstr.close()

       title, passage = self.clean_text(str)
       return title, passage

    # convert pdf file text to string and save as a text_pdf.txt file
    def save_convert_pdf_to_txt(self):
       content = self.convert_pdf_to_txt()
       txt_pdf = open('text_pdf.txt', 'wb')
       txt_pdf.write(content.encode('utf-8'))
       txt_pdf.close()

    def clean_text(self, text):
        ## split title and passage
        text_split = text.split('\n\n')
        title = text_split[0].replace('\n', ' ')
        passage = "\n\n".join(text_split[1:])
        ## split sentences and filter noise in passage
        passage_split = passage.split('\n')
        id_kept = []
        for cur_id in range(len(passage_split)):
            cur_line = passage_split[cur_id]
            # not including references
            if cur_line.lower() == "REFERENCES".lower():
                break
            # not including some noise lines
            # if len(cur_line.split(" ")) >= 4:
            if len(cur_line) >= 3:
                id_kept.append(cur_id)
        passage_kept = [passage_split[id] for id in id_kept]
        passage_kept = ' '.join(passage_kept)
        # "mes- sages"
        passage_kept = passage_kept.replace("- ", "")
        return title, passage_kept



if __name__ == '__main__':
    root_data_dir = "./Data/Surveys/"
    file_list = os.listdir(root_data_dir)
    for f in file_list:
        file_path = os.path.join(root_data_dir, f)
        pdfConverter = PdfConverter(file_path=file_path)
        title, passage = pdfConverter.convert_pdf_to_txt()
        # print("passage: ", passage)
        print("title: ", title)
