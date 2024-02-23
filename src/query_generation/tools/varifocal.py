def extract_focals(nlp, text):
        print("Extracting focal points from claim")
        doc = nlp(text)
        sentences = doc.sents

        tag_set = {
            "clause_relations": ["csubj", "xcomp", "ccomp", "advcl", "acl"],
            "sentence_core": ["nsubj", "obj", "iobj"],
            "full_syntax": ["csubj", "xcomp", "ccomp", "advcl", "acl", "nsubj", "obj", "iobj"],
            "noun_modifiers": ["nmod", "amod", "obj", "num"],
            "phrase_details": ["nsubj", "obj", "iobj", "nmod", "amod", "obj", "num"],
            "syntax_all": ["csubj", "xcomp", "ccomp", "advcl", "acl", "nsubj", "obj", "iobj", "nmod", "amod", "obj", "num"],
            "sub_verb_obj": ["agent", "ccomp", "csubj", "dobj", "nsubj", "nsubjpass", "pcomp", "pobj", "root"]
        }

        active_tag_set = tag_set["sub_verb_obj"]
        subtrees = []

        for sentence in sentences:
            for token in sentence:
                if token.dep_ in active_tag_set:
                    subtree = [t.text for t in token.subtree]
                    subtrees.append([" ".join(subtree), token.dep_])

        print("Focal points:", subtrees, "extracted.")
        return subtrees
    
def generate_questions(pipe, text, focal_points):
    print("Generating questions from focal points")
    questions = []

    for focal_point in focal_points:
        focal_text = focal_point[0]
        pipe_string = "answer: " + str(focal_text) + " context: " + str(text)

        output = pipe(pipe_string)
        
        question = output[0]['generated_text'].split("question: ")[1]
        questions.append(question)
    
    print("Questions generated:", questions)
    return questions

def reranking(questions):
    print("Reranking questions")
    return questions