class Intentrouter:
    def __init__(self,metadata) :
        self.metadata=metadata
    
    def intention(self,parsed_query):
        columns=parsed_query['columns']
        actions=parsed_query['actions']
        plan=[]

        if any(a in ['mean','sum','max','min'] for a in actions):
            plan.append('calculate_stats')
        
        if 'plot' in actions:
            plan.append('visualization')

        if 'why' in actions:
            plan.append('explainable_ai')

        if 'predict' in actions:
            plan.append('prediction')
        
        return plan
    
        