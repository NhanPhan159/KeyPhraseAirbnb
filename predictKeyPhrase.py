from preprocessing import predict,preprocess_text

def predictKey(text):
    text_pre = preprocess_text(text)
    keys = predict(text_pre)
    return keys