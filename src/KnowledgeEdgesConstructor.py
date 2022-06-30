class KnowledgesEdgesConstructor():
    
    def __init__(self, specifier_list):
        self.specifier_list = specifier_list
        self.Aes = {}
        
        
    def __define_nodes_informations(self, annotation, parameters):
               
        for specifier in self.specifier_list:
            
            Ae = specifier.define_Ae_knowledge(annotation, parameters)
            self.Aes[specifier.name] = Ae

    def get_knowledges(self, annotation, parameters):
        
        self.__define_nodes_informations(annotation, parameters)
        
        return self.Aes