#from subject import Subject

from .subject import Subject

class Subjects:
   
    def __init__(self, list_subjects): 
        self.list_subjects = list_subjects
        self.nb_subjects = len(list_subjects)

    def get_number_subjects(self):
        return self.nb_subjects

    def add_subject(self, subject : Subject):
        self.list_subjects.append(subject)
        self.nb_subjects = len(self.list_subjects)