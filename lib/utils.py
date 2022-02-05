from scipy import spatial


"""
Cosine similarity function for vectors
"""
def cosine_similarity(vector1, vector2):
    # return 1 - spatial.distance.cdist(vector1, vector2, 'cosine')
    return 1 - spatial.distance.cosine(vector1, vector2)
