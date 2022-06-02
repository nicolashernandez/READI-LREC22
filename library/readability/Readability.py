#Pseudo-docstring:
# The Readability class interacts with the library's modules to provide a bunch of useful functions / reproduce the paper's contents

# It is meant to provide the following :
# 1) At start-up : it "compiles" a text into a structure useful for the other functions, and also calculates a bunch of relevant statistics (number of words, sentences, syllables, etc..)
# 2) Access the relevant "simple" scores by using these pre-calculated statistics
# 3) Perform lazy loading of heavier stuff, like calculating perplexity and using models.
# 4) Access the things available in the other files.
#

class Readability:
    def __init__(self, lang = "fr", nlp_processor = "spacy_sm", perplexity_processor = "gpt2"):
        print("hello world my first parameter is", lang)
        # lang = fr, nlp_processor = spacy_sm, perplexity_processor = gpt2 truc... default values.
        #V0 will download everything at once when called.
        #V1 could implement lazy loading for the heavy stuff, like using a transformer model.
        #1) Compile text/corpora into relevant format

        #2) Prepare statistics

        #3) Load the "small" or local stuff like spacy"


        #4) Prepare ways to lazy load heavier stuff
        return 0
