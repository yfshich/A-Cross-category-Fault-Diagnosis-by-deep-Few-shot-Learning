import numpy.random as rng
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import os,json

from sys import stdout
def flush(string):
    stdout.write('\r')
    stdout.write(str(string))
    stdout.flush()


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,X_train,y_train,X_val,y_val):
        self.data = {'train':X_train,'val':X_val}
        self.labels = {'train':y_train,'val':y_val}
        train_classes = list(set(y_train))
        np.random.seed(10)
#         train_classes = sorted(rng.choice(train_classes,size=(int(len(train_classes)*0.8),),replace=False) )
        self.classes = {'train':sorted(train_classes),'val':sorted(list(set(y_val)))}
        self.indices = {'train':[np.where(y_train == i)[0] for i in self.classes['train']],
                        'val':[np.where(y_val == i)[0] for i in self.classes['val']]
                       }
        #print('================self.classes==================')
        #print(self.classes)
        #print('================self.indices==================')
        #print(self.indices)
        #print('================len(X_train),len(X_val)==================')
        #print(len(X_train),len(X_val))
        #print('================[len(c) for c in self.indices[train]],[len(c) for c in self.indices[val]]==================')
        #print([len(c) for c in self.indices['train']],[len(c) for c in self.indices['val']])
        
    def set_val(self,X_val,y_val):
        self.data['val'] = X_val
        self.labels['val'] = y_val
        self.classes['val'] =  sorted(list(set(y_val)))
        self.indices['val'] =  [np.where(y_val == i)[0] for i in self.classes['val']]
        

    def get_batch(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        n_classes = len(self.classes[s])
        #print('==============n_classes=================')
        #print(n_classes)
        X_indices = self.indices[s]
        #print('==============X_indices=================')
        #print(X_indices.shape)
        _, w, h = X.shape
#         if batch_size > n_classes:
#             raise ValueError("{} batch_size has greter than {} classes".format(batch_size,n_classes))

        #randomly sample several classes to use in the batch
        categories = rng.choice(n_classes,size=(batch_size,),replace=True)
        #print('==============categories=================')
        #print(categories)
        #initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((batch_size, w,h,1)) for i in range(2)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets=np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        #距离损失权重因子
        V = []
        for i in range(batch_size):
            if(i<batch_size//2):
                V.append(0.2)
            else:
                V.append(1)
        V = np.array(V)
        #print('=======================create pairs============================')
        for i in range(batch_size):
            category = categories[i]
            #print('=======================category============================')
            #print(category)
            n_examples = len(X_indices[category])
            #print('=======================n_examples============================')
            #print(n_examples)
            if(n_examples==0):
                print("error:n_examples==0",n_examples)
            #在python中的random.randint(a,b)用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b。
            idx_1 = rng.randint(0, n_examples)
            
            pairs[0][i,:,:,:] = X[X_indices[category][idx_1]].reshape(w, h, 1)
            #pick images of same class for 1st half, different for 2nd
            #这部分是制造成对的输入，即，对输入的batch——seize改造前半部分是相同的类别，后半部分是不同的类别
            if i >= batch_size // 2:
                category_2 = category  
                idx_2 = (idx_1 + rng.randint(1,n_examples)) % n_examples
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1,n_classes)) % n_classes
                n_examples = len(X_indices[category_2])
                idx_2 = rng.randint(0, n_examples)
            pairs[1][i,:,:,:] = X[X_indices[category_2][idx_2]].reshape(w, h,1)
        return pairs, targets, V, categories
        #return pairs, targets, categories
    
    #creat by yfshi 2019.12.16,加入齿轮箱数据，并将齿轮箱数据所占比例设为参数
    def get_batch_gear(self,batch_size,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        n_classes = len(self.classes[s])
        #print('==============n_classes=================')
        #print(n_classes)
        X_indices = self.indices[s]
        #print('==============X_indices=================')
        #print(X_indices.shape)
        _, w, h = X.shape
#         if batch_size > n_classes:
#             raise ValueError("{} batch_size has greter than {} classes".format(batch_size,n_classes))

        #randomly sample several classes to use in the batch
        categories = rng.choice(n_classes,size=(batch_size,),replace=True)
        #print('==============categories=================')
        #print(categories)
        #initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((batch_size, w,h,1)) for i in range(2)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets=np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        #print('=======================create pairs============================')
        for i in range(batch_size):
            category = categories[i]
            #print('=======================category============================')
            #print(category)
            n_examples = len(X_indices[category])
            #print('=======================n_examples============================')
            #print(n_examples)
            if(n_examples==0):
                print("error:n_examples==0",n_examples)
            #在python中的random.randint(a,b)用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b。
            idx_1 = rng.randint(0, n_examples)
            
            pairs[0][i,:,:,:] = X[X_indices[category][idx_1]].reshape(w, h, 1)
            #pick images of same class for 1st half, different for 2nd
            #这部分是制造成对的输入，即，对输入的batch——seize改造前半部分是相同的类别，后半部分是不同的类别
            if i >= batch_size // 2:
                category_2 = category  
                idx_2 = (idx_1 + rng.randint(1,n_examples)) % n_examples
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1,n_classes)) % n_classes
                n_examples = len(X_indices[category_2])
                idx_2 = rng.randint(0, n_examples)
            pairs[1][i,:,:,:] = X[X_indices[category_2][idx_2]].reshape(w, h,1)
        return pairs, targets, categories
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    

    def make_oneshot_task(self,N,s="val",language=None):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        n_classes = len(self.classes[s])
        X_indices = self.indices[s]
        _, w, h = X.shape
        if N > n_classes:
            raise ValueError("{} way task has greter than {} classes".format(N,n_classes))

        categories = rng.choice(n_classes,size=(N,),replace=False)            
        true_category = categories[0]
        n_examples = len(X_indices[true_category]) 
        ex1, ex2 = rng.choice(n_examples,size=(2,),replace=False)
        test_image = np.asarray([X[X_indices[true_category][ex1]]]*N).reshape(N, w, h,1)
        support_set = np.zeros((N,w,h))
        support_set[0,:,:] = X[X_indices[true_category][ex2]]
        for idx,category in enumerate(categories[1:]):
            n_examples = len(X_indices[category])
            support_set[idx+1,:,:] = X[X_indices[category][rng.randint(0,n_examples)]]
        support_set = support_set.reshape(N, w, h,1)
        targets = np.zeros((N,))
        #print("===============targets-before================")
        #print(targets)
        targets[0] = 1
        #print("===============targets-after================")
        #print(targets)
        targets, test_image, support_set,categories = shuffle(targets, test_image, support_set, categories)
        pairs = [test_image,support_set]

        return pairs, targets,categories
    
    def test_oneshot(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        val_c = self.labels[s]
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        preds = []
        err_print_num = 0
        for idx in range(k):
            inputs, targets,categories = self.make_oneshot_task(N,s)
#             print('=========================test_oneshot====================')
#             print(inputs)
#             print(targets)
#             print(categories)
#             print('=========================test_oneshot====================')
            n_classes, w, h,_ = inputs[0].shape
#             inputs[0]=inputs[0].reshape(n_classes,100,100,h)
#             inputs[1]=inputs[1].reshape(n_classes,100,100,h)
            inputs[0]=inputs[0].reshape(n_classes,w,h)
            inputs[1]=inputs[1].reshape(n_classes,w,h)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
            elif verbose and err_print_num<1:
                err_print_num = err_print_num +1
                #print(targets)
#                 print(categories)
#                 print('==========[categories[np.argmax(targets)],categories[np.argmax(probs)]]========')
#                 print([categories[np.argmax(targets)],categories[np.argmax(probs)]])
                inputs[0]=inputs[0].reshape(n_classes,w,h,1)
                inputs[1]=inputs[1].reshape(n_classes,w,h,1)
                plot_pairs(inputs,[np.argmax(targets),np.argmax(probs)])
            preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
#             preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct,preds

    def make_oneshot_task2(self,idx,s="val"):
        """Create pairs_list of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        X_labels = self.labels[s]
        
        X_train=self.data['val']
        X_train = X_train.reshape(len(X),2048,1)
        #print(X_train.shape)
        indices_train = self.indices['val']
        classes_train = self.classes['val']
        N = len(indices_train)
        #print(N)
        X = X.reshape(len(X),2048,1)
        #print(X.shape)
        _, w, h = X.shape
        #_, w, h = X.shape
        
        test_image = np.asarray([X[idx]]*N).reshape(N, w, h,1)
        support_set = np.zeros((N,w,h))
        #print(support_set.shape)
        #print(indices_train)
        for index in range(N):
            support_set[index,:,:] = X_train[rng.choice(indices_train[index],size=(1,),replace=False)]
        support_set = support_set.reshape(N, w, h,1)

        targets = np.zeros((N,))
        true_index = classes_train.index(X_labels[idx])
        targets[true_index] = 1
        
#         targets, test_image, support_set,categories = shuffle(targets, test_image, support_set, classes_train)
        categories = classes_train
     
        pairs = [test_image,support_set]
        
        return pairs, targets,categories
            
    def test_oneshot2(self,model,N,k,s="val",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        k = len(self.labels[s])
        #print(k)
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        preds = []
        probs_all = []
        err_print_num = 0
        for idx in range(k):
            inputs, targets,categories = self.make_oneshot_task2(idx,s)
#             print('=========================test_oneshot2====================')
#             print(inputs)
#             print(targets)
#             print(categories)
#             print('=========================test_oneshot2====================')
            n_classes, w, h,_ = inputs[0].shape
            inputs[0]=inputs[0].reshape(n_classes,2048,1)
            inputs[1]=inputs[1].reshape(n_classes,2048,1)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
            elif verbose and err_print_num<1:
                err_print_num = err_print_num +1
                #print(targets)
#                 print(categories)
                #print([categories[np.argmax(targets)],categories[np.argmax(probs)]])
                inputs[0]=inputs[0].reshape(n_classes,2048,1,1)
                inputs[1]=inputs[1].reshape(n_classes,2048,1,1)
                plot_pairs(inputs,[np.argmax(targets),np.argmax(probs)])
            preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
            probs_all.append(probs)
#             preds.append([categories[np.argmax(targets)],categories[np.argmax(probs)]])
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct,np.array(preds),np.array(probs_all)
    
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),)
# 20202.2.23 修改距离因子def train_and_test_oneshot(settings,siamese_net,siamese_loader,exp,time_idx):
def train_and_test_oneshot(settings,siamese_net,siamese_loader,exp,time_idx):
    settings['best'] = -1 
    settings['n'] = 0
    #print(settings)

    weights_path = settings["save_path"] + settings['save_weights_file']
    # if os.path.isfile(weights_path):
    #     print("load_weights",weights_path)
    #     siamese_net.load_weights(weights_path)
    print("training...")
    
    #Training loop
    losses = []
    accs = []
    for i in range(settings['n'], settings['n_iter']):
        #20191226距离因子
        #(inputs,targets, _)=siamese_loader.get_batch(settings['batch_size'])
        (inputs,targets, V, _)=siamese_loader.get_batch(settings['batch_size'])
        n_classes, w, h,_ = inputs[0].shape
    #     inputs[0]=inputs[0].reshape(n_classes,100,100,h)
    #     inputs[1]=inputs[1].reshape(n_classes,100,100,h)
    #     print(inputs[0].shape)
        inputs[0]=inputs[0].reshape(n_classes,w,h)
        inputs[1]=inputs[1].reshape(n_classes,w,h)
    #     print(inputs[0].shape)
        #20191226距离因子
        loss=siamese_net.train_on_batch(inputs,targets, V)
        
        #2020.2.13距离因子
#         losses.append(loss)
#         #loss=siamese_net.train_on_batch(inputs,targets)
#         val_acc, preds,probs_all= siamese_loader.test_oneshot2(siamese_net,settings['N_way'],settings['n_val'],verbose=False)
#         accs.append(val_acc)
#2020.2.13距离因子
        if i % settings['evaluate_every'] == 0:
            val_acc, preds,probs_all= siamese_loader.test_oneshot2(siamese_net,settings['N_way'],settings['n_val'],verbose=False)
            preds = np.array(preds)
            
            if val_acc >= settings['best'] :
                print("\niteration {} evaluating: {}".format(i,val_acc))
#                 print(loader.classes)
#                 score(preds[:,1],preds[:,0])
#                 print("\nsaving")
                siamese_net.save(weights_path)
                settings['best'] = val_acc
                settings['n'] = i
                with open(os.path.join(weights_path+".json"), 'w') as f:
                    f.write(json.dumps(settings, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': ')))

        if i % settings['loss_every'] == 0:
            flush("{} : {:.5f},".format(i,loss))
            losses.append(loss)
            # 前面省略，从下面直奔主题，举个代码例子：
            #Iresult2txt=str(i) 
            Lresult2txt=str(loss)          # data是前面运行出的数据，先将其转为字符串才能写入
            with open('./Loss_result/result'+str(exp)+'_'+str(time_idx)+'.txt','a') as file_handle:   # .txt可以不自己新建,代码会自动新建
                file_handle.write(Lresult2txt)     # 写入
                file_handle.write('\n')         # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
    np.save('./Loss_result/D=0W=0.5resultpearson_r'+str(exp)+'_'+str(time_idx)+'.npy', losses, allow_pickle=True, fix_imports=True)
    plt.plot(losses) 
    plt.show()
#     t = settings['name']
#     np.save("./Loss_Acc/"+str(t)+"_"+str(exp)+"_"+str(time_idx)+'loss.npy',losses)
#     np.save("./Loss_Acc/"+str(t)+"_"+str(exp)+"_"+str(time_idx)+'acc.npy',accs)
#     plt.plot(losses) 
#     plt.show()
#     plt.plot(accs) 
#     plt.show()
    return settings['best']
#def trainacc(losses):
    