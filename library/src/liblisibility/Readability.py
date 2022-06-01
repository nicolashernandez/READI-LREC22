#Pseudo-docstring:
#The Readability class interacts with the library's modules to provide a bunch of useful functions / reproduce the paper's contents

class Readability:
    def __init__(self):
        print("hello world")
        # lang = fr, nlp_processor = spacy_sm, perplexity_processor = gpt2 truc... default values.
        #V0 will download everything at once when called.
        #V1 could implement lazy loading for the heavy stuff, like using a transformer model.
        return 0
