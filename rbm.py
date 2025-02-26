from util import *

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.01
        
        self.momentum = 0.7

        self.print_period = 5000
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def cd1(self,visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]

        for it in range(n_iterations):

            # pick a random mini-batch
            batch_indices = np.random.randint(0, n_samples, self.batch_size)
            v_0 = visible_trainset[batch_indices]

            # positive phase
            p_h0, h_0 = self.get_h_given_v(v_0)
            
            # negative phase: reconstruct v_1 from h_0
            p_v1, v_1 = self.get_v_given_h(h_0)

            # **Clamp label** portion if this is the top RBM
            if self.is_top:
                pen_dim = self.ndim_visible - self.n_labels
                # Overwrite the last columns with the label bits from v_0
                v_1[:, pen_dim:]   = v_0[:, pen_dim:]
                p_v1[:, pen_dim:]  = v_0[:, pen_dim:]

            # final step: h_1 from v_1
            p_h1, h_1 = self.get_h_given_v(v_1)

            # param update
            self.update_params(v_0, h_0, v_1, h_1)

            # visualize once in a while if this RBM is at the bottom
            if it % self.rf["period"] == 0 and self.is_bottom:
                # show a few receptive fields
                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape(
                            (self.image_size[0], self.image_size[1], -1)),
                       it=it, grid=self.rf["grid"]
                )

            # print reconstruction loss occasionally
            if it % self.print_period == 0:
                recon_error = np.mean((v_0 - p_v1)**2)
                print(f"iteration={it:7d} recon_loss={recon_error:.4f}")
        
        return
    

    def update_params(self, v_0, h_0, v_k, h_k):
        batch_size = v_0.shape[0]

        # positive gradient
        pos_grad = np.dot(v_0.T, h_0) / batch_size  # shape (784, 200)
        # negative gradient
        neg_grad = np.dot(v_k.T, h_k) / batch_size  # shape (784, 200)
        grad_w = pos_grad - neg_grad

        # weight update
        self.delta_weight_vh = self.momentum * self.delta_weight_vh \
                            + self.learning_rate * grad_w
        self.weight_vh       += self.delta_weight_vh

        # bias_v update
        grad_bv = np.mean(v_0 - v_k, axis=0)  # shape (784,)
        self.delta_bias_v = self.momentum * self.delta_bias_v \
                            + self.learning_rate * grad_bv
        self.bias_v       += self.delta_bias_v

        # bias_h update
        grad_bh = np.mean(h_0 - h_k, axis=0)  # shape (200,)
        self.delta_bias_h = self.momentum * self.delta_bias_h \
                            + self.learning_rate * grad_bh
        self.bias_h       += self.delta_bias_h
        return

    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.1] hidden pre-activation
        support = np.dot(visible_minibatch, self.weight_vh) + self.bias_h
        p_h_given_v = sigmoid(support)
        h_sample    = sample_binary(p_h_given_v)
        return p_h_given_v, h_sample

    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:
            pen_dim = self.ndim_visible - self.n_labels  # e.g. 500 if pen=500, plus 10 labels => 510 total
            support = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v

            support_pen   = support[:, :pen_dim]               # first part: pen units
            support_label = support[:, pen_dim:]               # last part: label bits

            p_pen   = sigmoid(support_pen)
            pen_sam = sample_binary(p_pen)

            # for simplicity, treat label bits as Bernoulli
            p_label   = sigmoid(support_label)
            label_sam = sample_binary(p_label)

            # combine them back
            p_v_given_h = np.hstack([p_pen, p_label])
            v_sample    = np.hstack([pen_sam, label_sam])
            return p_v_given_h, v_sample
        
        else:
            # old code for the normal (non-top) case
            support = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v
            p_v_given_h = sigmoid(support)
            v_sample    = sample_binary(p_v_given_h)
            return p_v_given_h, v_sample

    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        n_samples = visible_minibatch.shape[0]
        # [TODO TASK 4.2] same as get_h_given_v but with self.weight_v_to_h
        support = np.dot(visible_minibatch, self.weight_v_to_h) + self.bias_h
        p_h_given_v = sigmoid(support)
        h_sample = sample_binary(p_h_given_v)
        return p_h_given_v, h_sample


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:
            # [TODO TASK 4.2 or 4.3]
            # Typically won't be used for the top RBM in standard DBN feedforward.
            raise ValueError("Should not call get_v_given_h_dir() on top RBM in normal DBN usage!")
        else:
            support = np.dot(hidden_minibatch, self.weight_h_to_v) + self.bias_v
            p_v_given_h = sigmoid(support)
            v_sample = sample_binary(p_v_given_h)
            return p_v_given_h, v_sample
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        # [TODO TASK 4.3] standard gradient is (trgs - preds)*inps
        batch_size = inps.shape[0]
        grad_w = np.dot(inps.T, (trgs - preds)) / batch_size
        grad_b = np.mean(trgs - preds, axis=0)

        self.delta_weight_h_to_v = self.momentum * self.delta_weight_h_to_v \
                                   + self.learning_rate * grad_w
        self.delta_bias_v        = self.momentum * self.delta_bias_v \
                                   + self.learning_rate * grad_b

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v        += self.delta_bias_v
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        batch_size = inps.shape[0]
        grad_w = np.dot(inps.T, (trgs - preds)) / batch_size
        grad_b = np.mean(trgs - preds, axis=0)

        self.delta_weight_v_to_h = self.momentum * self.delta_weight_v_to_h \
                                   + self.learning_rate * grad_w
        self.delta_bias_h        = self.momentum * self.delta_bias_h \
                                   + self.learning_rate * grad_b

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h        += self.delta_bias_h
        
        return    
