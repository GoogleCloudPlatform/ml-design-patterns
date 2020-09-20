
import tensorflow_hub as hub
swivel = hub.load("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1")

def preprocessing_fn(inputs):
    import tensorflow as tf
    
    outputs = inputs.copy()
    comments = inputs['comments']
    outputs['office'] = tf.strings.substr(comments, -4, 3)
    comments = tf.strings.regex_replace(comments, r"\([A-Z]+\)$", "")
    outputs['comments'] = tf.strings.lower(comments)
    #if len(outputs['comments'].shape) == 0:
    #    swivel_input = [outputs['comments']]
    #else:
    #    swivel_input = outputs['comments']
    #outputs['embed'] = swivel(swivel_input)
    return outputs
