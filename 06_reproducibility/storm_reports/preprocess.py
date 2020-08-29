def preprocessing_fn(inputs):
    import re
    
    outputs = inputs.copy()
    comments = inputs['comments']
    outputs['office'] = re.compile(r"\([A-Z]+\)$").search(comments).group(0)[1:-1]
    outputs['comments'] = comments[:-5].lower().strip()
    return outputs
