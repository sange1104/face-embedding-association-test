from tqdm import tqdm
import numpy as np

def forward(models, X, Y, AX, AY, BX, BY):
    model_names = models.keys()
    output = {}
    for i,model in tqdm(enumerate(models)):
        encode_X = model.predict(X)
        encode_Y = model.predict(Y)

        encode_AX = model.predict(AX)
        encode_AY = model.predict(AY)
        encode_BX = model.predict(BX)
        encode_BY = model.predict(BY)
        
        XY = np.concatenate([encode_X, encode_Y], axis=0) 
        XY = {i:XY[i] for i in range(len(XY))}

        _X = {i:X[i] for i in range(len(X))}
        _Y = {i+len(X):Y[i] for i in range(len(Y))}

        A = np.concatenate([encode_AX, encode_AY], axis=0) 
        B = np.concatenate([encode_BX, encode_BY], axis=0) 
        AB = np.concatenate([A, B], axis=0) 
        A = {i:A[i] for i in range(len(A))}
        B = {i+len(A):B[i] for i in range(len(B))}

        AB = {i:AB[i] for i in range(len(AB))}
        
        output[model_names[i]] = [XY, AB, _X, _Y, A, B]
        return output

def enc_to_dict(encode_X, encode_Y, encode_AX, encode_AY, encode_BX, encode_BY):
    encoded_XY = np.concatenate([encode_X, encode_Y], axis=0) 
    dict_XY = {i:encoded_XY[i] for i in range(len(encoded_XY))}

    dict_X = {i:X[i] for i in range(len(X))}
    dict_Y = {i+len(X):Y[i] for i in range(len(Y))}

    encoded_A = np.concatenate([encode_AX, encode_AY], axis=0) 
    encoded_B = np.concatenate([encode_BX, encode_BY], axis=0) 
    encoded_AB = np.concatenate([encoded_A, encoded_B], axis=0) 
    dict_A = {i:encoded_A[i] for i in range(len(encoded_A))}
    dict_B = {i+len(encoded_A):encoded_B[i] for i in range(len(encoded_B))}


    dict_AB = {i:encoded_AB[i] for i in range(len(encoded_AB))}
    return dict_XY, dict_AB, dict_X, dict_Y, dict_A, dict_B 