import numpy as np

def seq_chunk(seq,chunk,step=None,label=None):
    if step==None:
        step=chunk
    seq_length=seq.shape[0]
    rows = (seq_length - chunk) // step + 1
    new_shape=[]
    for i in range(len(seq.shape)):
        if i==0:
            new_shape.append(rows)
            new_shape.append(chunk)
        else:
            new_shape.append(seq.shape[i])
    new_set=np.empty(shape=new_shape)
    new_label = np.empty(shape=(new_shape[0],))
    for i in range(0,seq_length-chunk+1,step):
        new_set[i//step]=seq[i:i+chunk,:]
        new_label[i//step]=label
    return new_set,new_label

H_set=((100,200,300,400,600,900),(0,2,4,6,8,10))
I_set=((300,400,600,900),(0,2,4,6,8,10))
J_set=((400,600,900,1200),(0,2,4,6,8,10))
K_set=((600,900,1200,1500),(0,2,4,6,8,10))
All_set=(H_set,I_set,J_set,K_set)
Data=None
Label=None
for s,sets in enumerate(All_set):
    paths = ('Gear\\LW-00\\',
             'Gear\\LW-01\\',
             'Gear\\LW-02\\',
             'Gear\\LW-03\\',
             )
    for step, path in enumerate(paths):
        for i in sets[0]:
            for j in sets[1]:
                dir=path+str(i)+"-"+str(j)+".txt"
                tmp_datas,tmp_labels=seq_chunk(np.loadtxt(dir), 1024, 800, step)
                print(tmp_datas.shape,tmp_labels.shape)
                Data=tmp_datas if Data is None else np.concatenate((Data,tmp_datas),axis=0)
                Label = tmp_labels if Label is None else np.concatenate((Label ,tmp_labels), axis=0)
    np.save("gear_data2/set_"+str(s)+"_data.npy",Data)
    np.save("gear_data2/set_" + str(s) + "_label.npy",Label)
    Data = None
    Label = None