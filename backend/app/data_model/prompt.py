class Prompt:

    

    def __init__(self, id: str, prompt: str):
        self.id: str = id 
        self.prompt : str = prompt

    def get_prompt(self) -> str:

        return self.prompt
    
    def get_id(self) -> str:

        return self.id 