import pickle


filename="save_model/fps_14/memory"

fread=open(filename,'rb')
mem = pickle.load(fread)

print(len(mem))
# test merge 
