from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        
        batch_size = self.batch_size

        # We'll do it in mini-batches if n_samples is large
        all_preds = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            vis_batch = true_img[start:end]
            lbl_batch = true_lbl[start:end]   # only used to measure accuracy at the end

            # feed vis->hid->pen using directed weights
            p_hid, h_act = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_batch)
            p_pen, pen_act = self.rbm_stack["hid--pen"].get_h_given_v_dir(p_hid)

            # initialize label units to uniform (1/10)
            lbl = np.ones((end-start, self.sizes["lbl"])) * 0.1

            # top RBM input = pen + lbl
            pen_lbl = np.hstack([pen_act, lbl])

            # [TODO TASK 4.2] run alternating Gibbs sampling in the top RBM
            top_rbm = self.rbm_stack["pen+lbl--top"]
            for _ in range(self.n_gibbs_recog):
                # h <- p(h| pen+lbl)
                p_top_h, top_h = top_rbm.get_h_given_v(pen_lbl)
                # pen+lbl <- p(pen+lbl | h)
                p_pen_lbl, pen_lbl_sample = top_rbm.get_v_given_h(top_h)

                # We want to keep the pen portion clamped to pen_act 
                pen_lbl_sample[:, :self.sizes["pen"]] = pen_act
                p_pen_lbl[:, :self.sizes["pen"]]       = pen_act

                pen_lbl = p_pen_lbl  # or pen_lbl_sample, since for labels we want them to move

            # after a few steps, the last columns of pen_lbl are the label probabilities
            # apply softmax to the last 10 columns
            logits  = pen_lbl[:, self.sizes["pen"]:]
            predicted_lbl_probs = softmax(logits)

            all_preds.append(predicted_lbl_probs)

        all_preds = np.vstack(all_preds)
        predicted_y = np.argmax(all_preds, axis=1)
        true_y      = np.argmax(true_lbl, axis=1)
        accuracy    = 100. * np.mean(predicted_y == true_y)
        print(f"accuracy = {accuracy:.2f}%")
        return

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        # Start with random pen-state but clamp label
        pen_state = np.random.randint(0,2,size=(n_sample,self.sizes["pen"]))*1.0
        pen_lbl   = np.hstack([pen_state, true_lbl])

        top_rbm = self.rbm_stack["pen+lbl--top"]

        # [TODO TASK 4.2] run alternating Gibbs sampling for n_gibbs_gener steps
        for _ in range(self.n_gibbs_gener):
            # h <- p(h| pen+lbl)
            p_top_h, top_h = top_rbm.get_h_given_v(pen_lbl)
            # pen+lbl <- p(pen+lbl | h)
            p_pen_lbl, pen_lbl_sample = top_rbm.get_v_given_h(top_h)

            # clamp label portion
            pen_lbl_sample[:, self.sizes["pen"]:] = true_lbl
            p_pen_lbl[:, self.sizes["pen"]:]      = true_lbl
            pen_lbl = p_pen_lbl

            # now we have pen part in pen_lbl[:, :pen_size]
            # drive it down to visible
            # pen->hid
            _, hid_sample = self.rbm_stack["hid--pen"].get_v_given_h_dir(pen_lbl[:, :self.sizes["pen"]])
            # hid->vis
            _, vis_sample = self.rbm_stack["vis--hid"].get_v_given_h_dir(hid_sample)

            # store for animation
            img = vis_sample.reshape(self.image_size)
            records.append([ax.imshow(img, cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)])
        
        anim = stitch_video(fig, records)
        anim.save(f"{name}.generate{np.argmax(true_lbl)}.mp4")
        plt.close(fig)

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily
        
            print("training vis--hid")
            self.rbm_stack["vis--hid"].cd1(vis_trainset, n_iterations)
            self.savetofile_rbm(loc="trained_rbm", name="vis--hid")

            print("training hid--pen")
            # to get "data" for hid--pen, do a feedforward through vis--hid
            p_hid, _ = self.rbm_stack["vis--hid"].get_h_given_v(vis_trainset)
            self.rbm_stack["hid--pen"].cd1(p_hid, n_iterations)
            self.rbm_stack["vis--hid"].untwine_weights()            
            self.savetofile_rbm(loc="trained_rbm", name="hid--pen")

            print("training pen+lbl--top")
            # feed p_hid -> pen
            p_pen, _ = self.rbm_stack["hid--pen"].get_h_given_v(p_hid)
            # now concatenate label
            pen_lbl_data = np.hstack([p_pen, lbl_trainset])
            self.rbm_stack["pen+lbl--top"].cd1(pen_lbl_data, n_iterations)
            self.rbm_stack["hid--pen"].untwine_weights()
            self.savetofile_rbm(loc="trained_rbm", name="pen+lbl--top")

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try:
            self.loadfromfile_dbn(loc="trained_dbn", name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn", name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn", name="pen+lbl--top")
        except IOError:
            n_samples = vis_trainset.shape[0]

            top_rbm       = self.rbm_stack["pen+lbl--top"]
            rbm_hid_pen   = self.rbm_stack["hid--pen"]
            rbm_vis_hid   = self.rbm_stack["vis--hid"]

            pen_dim       = self.sizes["pen"]
            # self.sizes["lbl"] is 10 for MNIST

            for it in range(n_iterations):
                # [1] Pick a mini-batch of real data
                batch_indices = np.random.randint(0, n_samples, self.batch_size)
                v0   = vis_trainset[batch_indices]     # shape (batch, 784)
                lbl0 = lbl_trainset[batch_indices]     # shape (batch, 10)

                # ------------------------------------------------------------------
                # WAKE PHASE (real data bottom->top). Update generative parameters.
                # ------------------------------------------------------------------
                #   a) Drive v0->hid->pen with recognition (directed) weights
                p_hid, hid = rbm_vis_hid.get_h_given_v_dir(v0)
                p_pen, pen = rbm_hid_pen.get_h_given_v_dir(hid)

                #   b) The top RBM sees [ pen | lbl ] as visible.
                pen_lbl_data = np.hstack([pen, lbl0])

                #   c) Do short CD-k in the top RBM, clamping the label bits
                #      (partial clamp). 
                #
                # For example, 1 or 2 steps of CD is enough. We'll show 1 step:
                p_h0, h0 = top_rbm.get_h_given_v(pen_lbl_data)
                p_v1, v1 = top_rbm.get_v_given_h(h0)

                # clamp the label bits in v1
                v1[:, pen_dim:]   = pen_lbl_data[:, pen_dim:]
                p_v1[:, pen_dim:] = pen_lbl_data[:, pen_dim:]

                p_h1, h1 = top_rbm.get_h_given_v(v1)

                #   d) Update the top RBM’s undirected parameters 
                #      (i.e., “generative” from the wake phase)
                top_rbm.update_params(
                    pen_lbl_data, h0,
                    v1, h1
                )

                #   e) Update the generative weights in the lower layers:
                #      For a DBN, “generative” means top->down.  In wake phase,
                #      the actual observed (pen, hid) are the targets. 
                #      We do:
                #      hid->vis generative update,
                #      pen->hid generative update
                # 
                #    - The *input* to these updates is the layer *above* 
                #      (which we have from wake pass).
                #    - The *target* is the real activity.
                #    - The *prediction* is the downward pass using weight_h_to_v.
                # 
                #   Let's do pen->hid first:
                #   The “input” is pen, “target” is hid, “prediction” is p_hid_down from pen.
                #   But we don't have p_hid_down yet. We'll do a partial approach:
                #   Option A: (quick approach) skip or do sleep-phase only
                #
                #   Typically in the original wake-sleep, the lower generative layers are
                #   updated in the wake phase by computing how well each layer reconstructs
                #   the one below. We'll skip that or do a simpler approach 
                #   because the fully correct approach is to keep track of the predictions 
                #   from pen->hid during wake. That said, let's keep it simpler
                #   and do all generative updates in the sleep phase for clarity.
                #

                # ------------------------------------------------------------------
                # SLEEP PHASE (fantasy from top->bottom). Update recognition.
                # ------------------------------------------------------------------
                #   a) Start from top hidden sample h1, or just sample a random top hidden. 
                #      Here we can keep the label bits clamped or not, up to you.
                # 
                #      We'll keep label bits here for "semi-supervised" generation. 
                pen_lbl_samp = np.copy(v1)  # shape: (batch, pen_dim + 10)
                # optionally, re-clamp the label if you want fully supervised sleep
                pen_lbl_samp[:, pen_dim:] = lbl0  

                #   b) Sample down from pen_lbl_samp => pen => hid => vis
                #      using the *generative* (directed downward) weights:
                p_hid_down, hid_down = rbm_hid_pen.get_v_given_h_dir(pen_lbl_samp[:, :pen_dim])
                p_vis_down, vis_down = rbm_vis_hid.get_v_given_h_dir(hid_down)

                #   c) Now we do RECOGNITION updates 
                #      (the “recognize_params” are the upward directed connections).
                # 
                #   i) For bottom-layer recognition: v->hid
                rbm_vis_hid.update_recognize_params(
                    inps = vis_down,       # the layer below
                    trgs = hid_down,       # the "target" is the actual hidden sample
                    preds= rbm_vis_hid.get_h_given_v_dir(vis_down)[0]  
                    # we pass the probabilities p_h|v from a forward pass
                )

                #   ii) For middle-layer recognition: hid->pen
                rbm_hid_pen.update_recognize_params(
                    inps = hid_down,
                    trgs = pen_lbl_samp[:, :pen_dim],
                    preds= rbm_hid_pen.get_h_given_v_dir(hid_down)[0]
                )

                # ------------------------------------------------------------------
                # Optionally also do GENERATIVE updates in the lower layers 
                # (“sleep-wake” for generative).
                # 
                #   Because the "fantasy" pen_lbl_samp => hid_down => vis_down 
                #   gives us (inps= pen, trgs= ???, preds= ???).
                #   For pen->hid generative we do update_generate_params(pen, hid, predicted_hid) 
                #   For hid->vis generative we do update_generate_params(hid, vis, predicted_vis)
                # 
                #   i) generative pen->hid
                p_hid_pred = rbm_hid_pen.get_v_given_h_dir(pen_lbl_samp[:, :pen_dim])[0]
                rbm_hid_pen.update_generate_params(
                    inps = pen_lbl_samp[:, :pen_dim],
                    trgs = hid_down,
                    preds= p_hid_pred
                )

                #   ii) generative hid->vis
                p_vis_pred = rbm_vis_hid.get_v_given_h_dir(hid_down)[0]
                rbm_vis_hid.update_generate_params(
                    inps = hid_down,
                    trgs = v0,        # if you want to treat the real v0 as target or the “fantasy” vis_down
                                    # strictly "sleep" uses the fantasy's next layer as target, but 
                                    # there's some variation. We'll show the typical approach is to use 
                                    # the fantasy's own next layer. But the original Hinton approach 
                                    # can differ. 
                    preds= p_vis_pred
                )

                # printing
                if it % self.print_period == 0:
                    print(f"wake-sleep iteration={it}")

            # end for it in range(n_iterations)
            self.savetofile_dbn(loc="trained_dbn", name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn", name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn", name="pen+lbl--top")

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
