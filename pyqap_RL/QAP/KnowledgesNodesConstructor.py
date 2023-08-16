class KnowledgesNodesConstructor():
    
    def __init__(self, specifier_list):
        self.specifier_list = specifier_list
        self.Ans = {}
        
        
    def __define_nodes_informations(self, annotation, parameters = None):
               
        for specifier in self.specifier_list:
            
            An = specifier.define_Ar_knowledge(annotation)
            self.Ans[specifier.name] = An

    def get_knowledges(self, annotation, parameteres = None):
        
        self.__define_nodes_informations(annotation)
        
        return self.Ans