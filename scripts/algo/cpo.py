import tensorflow as tf
import numpy as np
import cPickle as pickle
from utils.common import *

#Code template from https://github.com/dennybritz/reinforcement-learning
#Code is heavily borrowed from public implementation of Constrained Policy Optimization - https://github.com/jachiam/cpo

class Actor():
#Define the actor(policy estimator) over here

    def get_fisher_product_op(self):
        directional_gradients = tf.reduce_sum(self.kl_flat_gradients_op*self.vec)
        return get_flat_gradients(directional_gradients, self.trainable_variables)

    def get_fisher_product(self, vec, damping = 1e-3):
        self.feed_dict[self.vec] = vec
        return self.sess.run(self.fisher_product_op, self.feed_dict) + damping*vec

    def __init__(self, n_states, action_dim, action_limits, action_log_stddevs, desired_kl, session, arch = 'asl', filename = None, summary = True):

        self.sess = session

        with tf.variable_scope("Actor"):

            with tf.name_scope("EpisodeData"):
                self.state = tf.placeholder(tf.float32, [None, n_states], name = 'States')
                self.action = tf.placeholder(tf.float32, [None, action_dim], name = 'Actions')
                self.advantage = tf.placeholder(tf.float32, [None], name = 'Advantages')
                self.safety_advantage = tf.placeholder(tf.float32, [None], name = 'SafetyAdvantages')
                self.safety_constraint = tf.placeholder(dtype = tf.float32, shape = [], name = 'SafetyConstraint')
                self.n_episodes = tf.placeholder(dtype = tf.float32, shape = [], name = 'NumberEpisodes')

            with tf.name_scope('Model'):
                self.train_iteration = 0
                self.n_input = n_states
                if(arch == 'asl'):
                    self.n_hidden1 = 1000
                    self.n_hidden2 = 300
                    self.n_hidden3 = 100
                elif(arch == 'tai'):
                    self.n_hidden1 = 512
                    self.n_hidden2 = 512
                    self.n_hidden3 = 512

                self.action_dim = action_dim
                self.n_output = action_dim

                if (filename is None):#Initialize Weights
                    with tf.name_scope('Weights'):
                        self.weights = {'h1' : tf.Variable(tf.random_normal([self.n_input, self.n_hidden1], stddev = tf.sqrt(2./self.n_input)), name = 'h1'), 'h2' : tf.Variable(tf.random_normal([self.n_hidden1, self.n_hidden2], stddev = tf.sqrt(2./self.n_hidden1)), name = 'h2'), 'h3' : tf.Variable(tf.random_normal([self.n_hidden2, self.n_hidden3], stddev = tf.sqrt(2./self.n_hidden2)), name = 'h3'), 'out' : tf.Variable(tf.random_normal([self.n_hidden3, self.n_output], stddev = tf.sqrt(2./self.n_hidden3)), name = 'out')}
                    with tf.name_scope('Biases'):
                        self.biases = {'b1' : tf.Variable(tf.zeros([self.n_hidden1]), name = 'b1'), 'b2' : tf.Variable(tf.zeros([self.n_hidden2]), name = 'b2'), 'b3' : tf.Variable(tf.zeros([self.n_hidden3]), name = 'b3'), 'out' : tf.Variable(tf.zeros([self.n_output]), name = 'out')}

                else:#Load Weights
                    data = pickle.load( open( filename, "rb" ) )
                    print("Initializing weights from '{}'".format(filename))
                    weights = data[0]
                    biases = data[1]
                    with tf.name_scope('Weights'):
                        self.weights = {'h1' : tf.Variable(weights['h1'], name = 'h1'), 'h2' : tf.Variable(weights['h2'], name = 'h2'), 'h3' : tf.Variable(weights['h3'], name = 'h3'), 'out' : tf.Variable(weights['out'], name = 'out')}
                    with tf.name_scope('Biases'):
                        self.biases = {'b1' : tf.Variable(biases['b1'], name = 'b1'), 'b2' : tf.Variable(biases['b2'], name = 'b2'), 'b3' : tf.Variable(biases['b3'], name = 'b3'), 'out' : tf.Variable(biases['out'], name = 'out')}

                with tf.name_scope('StandardDeviations'):
                    if(action_log_stddevs is None):
                        self.action_log_stddevs = tf.Variable(tf.zeros([self.n_output]), name = 'stddev')
                    else:
                        self.action_log_stddevs = tf.Variable(action_log_stddevs*tf.ones([self.n_output]), name = 'stddev')

                if(arch == 'asl'):
                    self.hidden_layer1 = tf.nn.tanh(tf.add(tf.matmul(self.state,  self.weights['h1']), self.biases['b1']))
                    self.hidden_layer2 = tf.nn.tanh(tf.add(tf.matmul(self.hidden_layer1, self.weights['h2']), self.biases['b2']))
                    self.hidden_layer3 = tf.nn.tanh(tf.add(tf.matmul(self.hidden_layer2, self.weights['h3']), self.biases['b3']))
                    self.output_layer = tf.nn.tanh(tf.add(tf.matmul(self.hidden_layer3, self.weights['out']), self.biases['out']))
                    action_list = []
                    for i in range(self.action_dim):
                        a = action_limits[i][0]
                        b = action_limits[i][1]
                        action_list.append(((b-a)*self.output_layer[:,i]/2 + (a+b)/2))

                elif(arch == 'tai'):
                    self.hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(self.state,  self.weights['h1']), self.biases['b1']))
                    self.hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_layer1, self.weights['h2']), self.biases['b2']))
                    self.hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(self.hidden_layer2, self.weights['h3']), self.biases['b3']))
                    self.output_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden_layer3, tf.reshape(self.weights['out'][:,0], (-1,1))), self.biases['out'][0]))
                    self.output_layer2 = tf.nn.tanh(tf.add(tf.matmul(self.hidden_layer3, tf.reshape(self.weights['out'][:,1], (-1,1))), self.biases['out'][1]))
                    self.output_layer = tf.concat([self.output_layer1, self.output_layer2], axis=1)
                    action_list = []
                    action_list.append(action_limits[0][1]*self.output_layer[:,0])
                    a = action_limits[1][0]
                    b = action_limits[1][1]
                    action_list.append(((b-a)*self.output_layer[:,1]/2 + (a+b)/2))

                self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
                self.trainable_variables_shapes = [var.get_shape().as_list() for var in self.trainable_variables]

                self.action_means = tf.stack(action_list, axis = 1)
                self.action_stddevs = tf.exp(self.action_log_stddevs) + 1e-8

                self.action_dist = tf.contrib.distributions.Normal(self.action_means, self.action_stddevs)

                self.sample_action = tf.squeeze(self.action_dist.sample())

                self.each_entropy_loss = tf.reduce_sum(self.action_dist.entropy(), axis = 1)
                self.average_entropy_loss = tf.reduce_mean(self.each_entropy_loss)

                self.action_log_probs = tf.reduce_sum(self.action_dist.log_prob(self.action), axis = 1)

                self.each_experience_loss = -self.action_log_probs * self.advantage
                self.average_experience_loss = tf.reduce_mean(self.each_experience_loss)

                self.chosen_action_log_probs = tf.reduce_sum(self.action_dist.log_prob(self.action), axis = 1)
                self.old_chosen_action_log_probs = tf.stop_gradient(tf.placeholder(tf.float32, [None]))
                self.each_safety_loss = tf.exp(self.chosen_action_log_probs - self.old_chosen_action_log_probs) * self.safety_advantage
                self.average_safety_loss = tf.reduce_sum(self.each_safety_loss)/self.n_episodes

                #May want to add regularization
                self.loss = self.average_experience_loss

                ##For Diagnostics
                self.old_action_means = tf.stop_gradient(tf.placeholder(tf.float32, [None, self.action_dim]))
                self.old_action_stddevs = tf.stop_gradient(tf.placeholder(tf.float32, [self.action_dim]))
                self.old_action_dist = tf.contrib.distributions.Normal(self.old_action_means, self.old_action_stddevs)

                self.each_kl_divergence = tf.reduce_sum(tf.contrib.distributions.kl_divergence(self.action_dist, self.old_action_dist), axis = 1)
                self.average_kl_divergence = tf.reduce_mean(self.each_kl_divergence)
                self.kl_gradients = tf.gradients(self.average_kl_divergence, self.trainable_variables)

                self.desired_kl = desired_kl
                self.metrics = [self.loss, self.average_kl_divergence, self.average_safety_loss]

                self.flat_params_op = get_flat_params(self.trainable_variables)
                self.loss_flat_gradients_op = get_flat_gradients(self.loss, self.trainable_variables)
                self.kl_flat_gradients_op = get_flat_gradients(self.average_kl_divergence, self.trainable_variables)
                self.constraint_flat_gradients_op = get_flat_gradients(self.average_safety_loss, self.trainable_variables)

                self.vec = tf.placeholder(tf.float32, [None])
                self.fisher_product_op = self.get_fisher_product_op()

                self.new_params = tf.placeholder(tf.float32, [None])
                self.params_assign_op = assign_network_params_op(self.new_params, self.trainable_variables, self.trainable_variables_shapes)


            if(summary):
                self._create_summaries()

    def _create_summaries(self):
        with tf.name_scope('Summaries'):
            self.summary = list()
            with tf.name_scope('Loss'):
                self.summary.append(tf.summary.histogram('each_experience_loss', self.each_experience_loss))
                self.summary.append(tf.summary.scalar('average_experience_loss', self.average_experience_loss))
                self.summary.append(tf.summary.histogram('each_entropy_loss', self.each_entropy_loss))
                self.summary.append(tf.summary.scalar('average_entropy_loss', self.average_entropy_loss))
                self.summary.append(tf.summary.histogram('each_kl_divergence', self.each_kl_divergence))
                self.summary.append(tf.summary.scalar('average_kl_divergence', self.average_kl_divergence))
            with tf.name_scope('Outputs'):
                self.summary.append(tf.summary.histogram('hidden_layer1', self.hidden_layer1))
                self.summary.append(tf.summary.histogram('hidden_layer2', self.hidden_layer2))
                self.summary.append(tf.summary.histogram('hidden_layer3', self.hidden_layer3))
                self.summary.append(tf.summary.histogram('output_layer', self.output_layer))
                #self.summary.append(tf.summary.histogram('output_layer_std', self.output_layer_std))
                for i in range(self.action_dim):
                    self.summary.append(tf.summary.histogram('action_means_'+str(i+1), self.action_means[:,i]))
                    self.summary.append(tf.summary.scalar('standard_deviation_'+str(i+1), self.action_stddevs[i]))
                    #self.summary.append(tf.summary.histogram('standard_deviation_'+str(i+1), self.action_stddevs[:,i]))
            with tf.name_scope('Weights'):
                self.summary.append(tf.summary.histogram('hidden_layer1', self.weights['h1']))
                self.summary.append(tf.summary.histogram('hidden_layer2', self.weights['h2']))
                self.summary.append(tf.summary.histogram('hidden_layer3', self.weights['h3']))
                self.summary.append(tf.summary.histogram('output_layer', self.weights['out']))
            with tf.name_scope('Biases'):
                self.summary.append(tf.summary.histogram('hidden_layer1', self.biases['b1']))
                self.summary.append(tf.summary.histogram('hidden_layer2', self.biases['b2']))
                self.summary.append(tf.summary.histogram('hidden_layer3', self.biases['b3']))
                self.summary.append(tf.summary.histogram('output_layer', self.biases['out']))

            self.summary_op = tf.summary.merge(self.summary)

    def update_weights(self, states, actions, advantages, safety_advantages, safety_constraint, n_episodes):
        self.train_iteration += 1
        self.feed_dict = { self.state: states, self.action: actions, self.advantage: advantages, self.safety_advantage: safety_advantages, self.n_episodes: n_episodes}
        chosen_action_log_probs =  self.sess.run(self.chosen_action_log_probs, self.feed_dict)
        self.feed_dict[self.old_chosen_action_log_probs] = chosen_action_log_probs
        g, b, old_action_means, old_action_stddevs, old_params, old_safety_loss = self.sess.run([self.loss_flat_gradients_op, self.constraint_flat_gradients_op, self.action_means, self.action_stddevs, self.flat_params_op, self.average_safety_loss], self.feed_dict)
        self.feed_dict[self.old_action_means] = old_action_means
        self.feed_dict[self.old_action_stddevs] = old_action_stddevs
        v = do_conjugate_gradient(self.get_fisher_product, g)
        #H_b = doConjugateGradient(self.getFisherProduct, b)
        approx_g = self.get_fisher_product(v)
        #b = self.getFisherProduct(H_b)
        linear_constraint_threshold = np.maximum(0, safety_constraint) + old_safety_loss
        eps = 1e-8
        delta = 2*self.desired_kl
        c = -safety_constraint
        q = np.dot(approx_g, v)

        if(np.dot(b,b) < eps):
            lam = np.sqrt(q/delta)
            nu = 0
            w = 0
            r,s,A,B = 0,0,0,0
            optim_case = 4
        else:
            norm_b = np.sqrt(np.dot(b,b))
            unit_b = b/norm_b
            w = norm_b * do_conjugate_gradient(self.get_fisher_product, unit_b)
            r = np.dot(w, approx_g)
            s = np.dot(w, self.get_fisher_product(w))
            A = q - (r**2/s)
            B = delta - (c**2/s)
            if (c < 0 and B < 0):
                optim_case = 3
            elif (c < 0 and B > 0):
                optim_case = 2
            elif(c > 0 and B > 0):
                optim_case = 1
            else:
                optim_case = 0
            lam = np.sqrt(q/delta)
            nu = 0

            if(optim_case == 2 or optim_case == 1):
                lam_mid = r / c
                L_mid = - 0.5 * (q / lam_mid + lam_mid * delta)

                lam_a = np.sqrt(A / (B + eps))
                L_a = -np.sqrt(A*B) - r*c / (s + eps)

                lam_b = np.sqrt(q / delta)
                L_b = -np.sqrt(q * delta)

                if lam_mid > 0:
                    if c < 0:
                        if lam_a > lam_mid:
                            lam_a = lam_mid
                            L_a   = L_mid
                        if lam_b < lam_mid:
                            lam_b = lam_mid
                            L_b   = L_mid
                    else:
                        if lam_a < lam_mid:
                            lam_a = lam_mid
                            L_a   = L_mid
                        if lam_b > lam_mid:
                            lam_b = lam_mid
                            L_b   = L_mid

                    if L_a >= L_b:
                        lam = lam_a
                    else:
                        lam = lam_b

                else:
                    if c < 0:
                        lam = lam_b
                    else:
                        lam = lam_a

                nu = max(0, lam * c - r) / (s + eps)

        if optim_case > 0:
            full_step = (1. / (lam + eps) ) * ( v + nu * w )
        else:
            full_step = np.sqrt(delta / (s + eps)) * w

        print('Optimization Case: ', optim_case)

        if(optim_case == 0 or optim_case == 1):
            new_params, status = do_line_search_CPO(self.get_metrics, old_params, full_step, self.desired_kl, linear_constraint_threshold, check_loss = False)
        else:
            new_params, status = do_line_search_CPO(self.get_metrics, old_params, full_step, self.desired_kl, linear_constraint_threshold)

        print('Success: ', status)

        if(status == False):
            self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})

        return old_action_means, old_action_stddevs

    def predict_action(self, states):
        return self.sess.run(self.sample_action, feed_dict = { self.state: states })

    def get_deterministic_action(self, states):
        return np.squeeze(self.sess.run(self.action_means, feed_dict = { self.state: states }))

    def get_summary(self, states, actions, advantages, safety_advantages, n_episodes, old_action_means, old_action_stddevs):
        feed_dict = { self.state: states, self.action: actions, self.advantage: advantages, self.safety_advantage: safety_advantages, self.n_episodes: n_episodes, self.old_action_means: old_action_means, self.old_action_stddevs: old_action_stddevs}
        return self.sess.run([self.summary_op, self.average_kl_divergence], feed_dict)

    def get_weights(self):
        return self.sess.run([self.weights, self.biases])

    def get_metrics(self, new_params):
        self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return self.sess.run(self.metrics, self.feed_dict)


class Critic():
#Define the critic(value function estimator) over here

    def get_fisher_product_op(self):
        directional_gradients = tf.reduce_sum(self.kl_flat_gradients_op*self.vec)
        return get_flat_gradients(directional_gradients, self.trainable_variables)

    def get_fisher_product(self, vec, damping = 1e-3):
        self.feed_dict[self.vec] = vec
        return self.sess.run(self.fisher_product_op, self.feed_dict) + damping*vec

    def __init__(self, n_states, desired_kl, session, arch = 'asl', filename = None, summary = True):

        self.sess = session

        with tf.variable_scope('Critic'):

            with tf.name_scope('EpisodeData'):
                self.state = tf.placeholder(tf.float32, [None, n_states], name = 'States')
                self.target = tf.placeholder(tf.float32, [None], name = "Targets")

            with tf.name_scope('Model'):
                self.train_iteration = 0
                self.n_input = n_states
                self.n_output = 1

                if (filename is None):#Initialize Weights
                    if(arch == 'asl'):
                        self.n_hidden1 = 1000
                        self.n_hidden2 = 300
                        self.n_hidden3 = 100
                    elif(arch == 'tai'):
                        self.n_hidden1 = 512
                        self.n_hidden2 = 512
                        self.n_hidden3 = 512

                    with tf.name_scope('Weights'):
                        self.weights = {'h1' : tf.Variable(tf.random_normal([self.n_input, self.n_hidden1], stddev = tf.sqrt(2./self.n_input)), name = 'h1'), 'h2' : tf.Variable(tf.random_normal([self.n_hidden1, self.n_hidden2], stddev = tf.sqrt(2./self.n_hidden1)), name = 'h2'), 'h3' : tf.Variable(tf.random_normal([self.n_hidden2, self.n_hidden3], stddev = tf.sqrt(2./self.n_hidden2)), name = 'h3'), 'out' : tf.Variable(tf.random_normal([self.n_hidden3, self.n_output], stddev = tf.sqrt(2./self.n_hidden3)), name = 'out')}
                    with tf.name_scope('Biases'):
                        self.biases = {'b1' : tf.Variable(tf.zeros([self.n_hidden1]), name = 'b1'), 'b2' : tf.Variable(tf.zeros([self.n_hidden2]), name = 'b2'), 'b3' : tf.Variable(tf.zeros([self.n_hidden3]), name = 'b3'), 'out' : tf.Variable(tf.zeros([self.n_output]), name = 'out')}

                else:#Load Weights
                    data = pickle.load( open( filename, "rb" ) )
                    weights = data[0]
                    biases = data[1]
                    with tf.name_scope('Weights'):
                        self.weights = {'h1' : tf.Variable(weights['h1'], name = 'h1'), 'h2' : tf.Variable(weights['h2'], name = 'h2'), 'h3' : tf.Variable(weights['h3'], name = 'h3'), 'out' : tf.Variable(weights['out'], name = 'out')}
                    with tf.name_scope('Biases'):
                        self.biases = {'b1' : tf.Variable(biases['b1'], name = 'b1'), 'b2' : tf.Variable(biases['b2'], name = 'b2'), 'b3' : tf.Variable(biases['b3'], name = 'b3'), 'out' : tf.Variable(biases['out'], name = 'out')}

                if(arch == 'asl'):
                    self.hidden_layer1 = tf.nn.tanh(tf.add(tf.matmul(self.state,  self.weights['h1']), self.biases['b1']))
                    self.hidden_layer2 = tf.nn.tanh(tf.add(tf.matmul(self.hidden_layer1, self.weights['h2']), self.biases['b2']))
                    self.hidden_layer3 = tf.nn.tanh(tf.add(tf.matmul(self.hidden_layer2, self.weights['h3']), self.biases['b3']))
                    self.output_layer = tf.add(tf.matmul(self.hidden_layer3, self.weights['out']), self.biases['out'])
                elif(arch == 'tai'):
                    self.hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(self.state,  self.weights['h1']), self.biases['b1']))
                    self.hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_layer1, self.weights['h2']), self.biases['b2']))
                    self.hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(self.hidden_layer2, self.weights['h3']), self.biases['b3']))
                    self.output_layer = tf.add(tf.matmul(self.hidden_layer3, self.weights['out']), self.biases['out'])

                self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
                self.trainable_variables_shapes = [var.get_shape().as_list() for var in self.trainable_variables]

                self.value_estimate = tf.squeeze(self.output_layer)

                self.variance = tf.placeholder(tf.float32, [], name = 'Variance')
                self.distribution = tf.contrib.distributions.Normal(self.value_estimate, self.variance)

                self.each_experience_loss = tf.pow(self.value_estimate - self.target, 2)
                self.average_experience_loss = tf.reduce_mean(self.each_experience_loss)

                #May want to add regularization
                self.loss = self.average_experience_loss

                ##For Diagnostics
                self.old_predicted_targets = tf.stop_gradient(tf.placeholder(tf.float32, [None]))
                self.old_distribution = tf.contrib.distributions.Normal(self.old_predicted_targets, self.variance)

                self.each_kl_divergence = tf.contrib.distributions.kl_divergence(self.old_distribution, self.distribution)
                self.average_kl_divergence = tf.reduce_mean(self.each_kl_divergence)

                self.target_variance = get_variance(self.target)
                self.explained_variance_before = 1 - get_variance(self.target-self.old_predicted_targets)/(self.target_variance + 1e-10)
                self.explained_variance_after = 1 - get_variance(self.target-tf.squeeze(self.value_estimate))/(self.target_variance + 1e-10)

                self.desired_kl = desired_kl
                self.metrics = [self.loss, self.average_kl_divergence]

                self.flat_params_op = get_flat_params(self.trainable_variables)
                self.loss_flat_gradients_op = get_flat_gradients(self.loss, self.trainable_variables)
                self.kl_flat_gradients_op = get_flat_gradients(self.average_kl_divergence, self.trainable_variables)

                self.vec = tf.placeholder(tf.float32, [None])
                self.fisher_product_op = self.get_fisher_product_op()

                self.new_params = tf.placeholder(tf.float32, [None])
                self.params_assign_op = assign_network_params_op(self.new_params, self.trainable_variables, self.trainable_variables_shapes)

            if(summary):
                self._createSummaries()

    def _createSummaries(self):
        with tf.name_scope('Summaries'):
            with tf.name_scope('Loss'):
                self.each_experience_loss_summary = tf.summary.histogram('each_experience_loss', self.each_experience_loss)
                self.average_experience_loss_summary = tf.summary.scalar('average_experience_loss', self.average_experience_loss)
                self.explained_variance_before_summary = tf.summary.scalar('explained_variance_before', self.explained_variance_before)
                self.explained_variance_after_summary = tf.summary.scalar('explained_variance_after', self.explained_variance_after)
            with tf.name_scope('Outputs'):
                self.hidden_layer1_summary = tf.summary.histogram('hidden_layer1', self.hidden_layer1)
                self.hidden_layer2_summary = tf.summary.histogram('hidden_layer2', self.hidden_layer2)
                self.hidden_layer3_summary = tf.summary.histogram('hidden_layer3', self.hidden_layer3)
                self.output_layer_summary = tf.summary.histogram('value_estimate', self.value_estimate)
            with tf.name_scope('Weights'):
                self.weights_hidden_layer1_summary = tf.summary.histogram('hidden_layer1', self.weights['h1'])
                self.weights_hidden_layer2_summary = tf.summary.histogram('hidden_layer2', self.weights['h2'])
                self.weights_hidden_layer3_summary = tf.summary.histogram('hidden_layer3', self.weights['h3'])
                self.weights_output_layer_summary = tf.summary.histogram('output_layer', self.weights['out'])
            with tf.name_scope('Biases'):
                self.biases_hidden_layer1_summary = tf.summary.histogram('hidden_layer1', self.biases['b1'])
                self.biases_hidden_layer2_summary = tf.summary.histogram('hidden_layer2', self.biases['b2'])
                self.biases_hidden_layer3_summary = tf.summary.histogram('hidden_layer3', self.biases['b3'])
                self.biases_output_layer_summary = tf.summary.histogram('output_layer', self.biases['out'])
            self.summary_op = tf.summary.merge([self.average_experience_loss_summary, self.each_experience_loss_summary, self.explained_variance_before_summary, self.explained_variance_after_summary, self.hidden_layer1_summary, self.hidden_layer2_summary, self.hidden_layer3_summary, self.output_layer_summary, self.weights_hidden_layer1_summary, self.weights_hidden_layer2_summary, self.weights_hidden_layer3_summary, self.weights_output_layer_summary, self.biases_hidden_layer1_summary, self.biases_hidden_layer2_summary, self.biases_hidden_layer3_summary, self.biases_output_layer_summary])


    def update_weights(self, states, targets):
        self.train_iteration += 1
        self.feed_dict = { self.state: states, self.target: targets  }
        loss, loss_gradients, old_predicted_targets, old_params = self.sess.run([self.loss, self.loss_flat_gradients_op, self.value_estimate, self.flat_params_op], self.feed_dict)
        self.feed_dict[self.old_predicted_targets] = old_predicted_targets
        self.feed_dict[self.variance] = loss
        step_direction = do_conjugate_gradient(self.get_fisher_product, loss_gradients)
        fisher_loss_product = self.get_fisher_product(step_direction)
        full_step_size = np.sqrt((2*self.desired_kl)/(np.dot(step_direction, fisher_loss_product)))
        full_step = full_step_size*step_direction
        #new_params = old_params + full_step
        new_params, status = do_line_search(self.get_metrics, old_params, full_step, self.desired_kl)
        if(status == False):
            self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return old_predicted_targets

    def predict_value(self, state):
        return self.sess.run(self.value_estimate, feed_dict = { self.state: state })

    def get_summary(self, states, targets, old_predicted_targets):
        feed_dict = { self.state: states, self.target: targets, self.old_predicted_targets: old_predicted_targets  }
        return self.sess.run(self.summary_op, feed_dict)

    def get_weights(self):
        return self.sess.run([self.weights, self.biases])

    def get_metrics(self, new_params):
        self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return self.sess.run(self.metrics, self.feed_dict)

class SafetyBaseline():
#Define the safety baseline estimator over here, similar to critic used above

    def get_fisher_product_op(self):
        directional_gradients = tf.reduce_sum(self.kl_flat_gradients_op*self.vec)
        return get_flat_gradients(directional_gradients, self.trainable_variables)

    def get_fisher_product(self, vec, damping = 1e-3):
        self.feed_dict[self.vec] = vec
        return self.sess.run(self.fisher_product_op, self.feed_dict) + damping*vec

    def __init__(self, n_states, desired_kl, session, arch = 'asl', filename = None, summary = True):

        self.sess = session

        with tf.variable_scope('SafetyBaseline'):

            with tf.name_scope('EpisodeData'):
                self.state = tf.placeholder(tf.float32, [None, n_states], name = 'States')
                self.target = tf.placeholder(tf.float32, [None], name = "Targets")

            with tf.name_scope('Model'):
                self.train_iteration = 0

                self.n_input = n_states
                self.n_output = 1

                if (filename is None):#Initialize Weights
                    if(arch == 'asl'):
                        self.n_hidden1 = 1000
                        self.n_hidden2 = 300
                        self.n_hidden3 = 100
                    elif(arch == 'tai'):
                        self.n_hidden1 = 512
                        self.n_hidden2 = 512
                        self.n_hidden3 = 512

                    with tf.name_scope('Weights'):
                        self.weights = {'h1' : tf.Variable(tf.random_normal([self.n_input, self.n_hidden1], stddev = tf.sqrt(2./self.n_input)), name = 'h1'), 'h2' : tf.Variable(tf.random_normal([self.n_hidden1, self.n_hidden2], stddev = tf.sqrt(2./self.n_hidden1)), name = 'h2'), 'h3' : tf.Variable(tf.random_normal([self.n_hidden2, self.n_hidden3], stddev = tf.sqrt(2./self.n_hidden2)), name = 'h3'), 'out' : tf.Variable(tf.random_normal([self.n_hidden3, self.n_output], stddev = tf.sqrt(2./self.n_hidden3)), name = 'out')}
                    with tf.name_scope('Biases'):
                        self.biases = {'b1' : tf.Variable(tf.zeros([self.n_hidden1]), name = 'b1'), 'b2' : tf.Variable(tf.zeros([self.n_hidden2]), name = 'b2'), 'b3' : tf.Variable(tf.zeros([self.n_hidden3]), name = 'b3'), 'out' : tf.Variable(tf.zeros([self.n_output]), name = 'out')}

                else:#Load Weights
                    data = pickle.load( open( filename, "rb" ) )
                    weights = data[0]
                    biases = data[1]
                    with tf.name_scope('Weights'):
                        self.weights = {'h1' : tf.Variable(weights['h1'], name = 'h1'), 'h2' : tf.Variable(weights['h2'], name = 'h2'), 'h3' : tf.Variable(weights['h3'], name = 'h3'), 'out' : tf.Variable(weights['out'], name = 'out')}
                    with tf.name_scope('Biases'):
                        self.biases = {'b1' : tf.Variable(biases['b1'], name = 'b1'), 'b2' : tf.Variable(biases['b2'], name = 'b2'), 'b3' : tf.Variable(biases['b3'], name = 'b3'), 'out' : tf.Variable(biases['out'], name = 'out')}

                if(arch == 'asl'):
                    self.hidden_layer1 = tf.nn.tanh(tf.add(tf.matmul(self.state,  self.weights['h1']), self.biases['b1']))
                    self.hidden_layer2 = tf.nn.tanh(tf.add(tf.matmul(self.hidden_layer1, self.weights['h2']), self.biases['b2']))
                    self.hidden_layer3 = tf.nn.tanh(tf.add(tf.matmul(self.hidden_layer2, self.weights['h3']), self.biases['b3']))
                    self.output_layer = tf.add(tf.matmul(self.hidden_layer3, self.weights['out']), self.biases['out'])
                elif(arch == 'tai'):
                    self.hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(self.state,  self.weights['h1']), self.biases['b1']))
                    self.hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(self.hidden_layer1, self.weights['h2']), self.biases['b2']))
                    self.hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(self.hidden_layer2, self.weights['h3']), self.biases['b3']))
                    self.output_layer = tf.add(tf.matmul(self.hidden_layer3, self.weights['out']), self.biases['out'])

                self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SafetyBaseline')
                self.trainable_variables_shapes = [var.get_shape().as_list() for var in self.trainable_variables]

                self.value_estimate = tf.squeeze(self.output_layer)

                self.variance = tf.placeholder(tf.float32, [], name = 'Variance')
                self.distribution = tf.contrib.distributions.Normal(self.value_estimate, self.variance)

                self.each_experience_loss = tf.pow(self.value_estimate - self.target, 2)
                self.average_experience_loss = tf.reduce_mean(self.each_experience_loss)

                #May want to add regularization
                self.loss = self.average_experience_loss

                ##For Diagnostics
                self.old_predicted_targets = tf.stop_gradient(tf.placeholder(tf.float32, [None]))
                self.old_distribution = tf.contrib.distributions.Normal(self.old_predicted_targets, self.variance)

                self.each_kl_divergence = tf.contrib.distributions.kl_divergence(self.old_distribution, self.distribution)
                self.average_kl_divergence = tf.reduce_mean(self.each_kl_divergence)

                self.target_variance = get_variance(self.target)
                self.explained_variance_before = 1 - get_variance(self.target-self.old_predicted_targets)/(self.target_variance + 1e-10)
                self.explained_variance_after = 1 - get_variance(self.target-tf.squeeze(self.value_estimate))/(self.target_variance + 1e-10)

                self.desired_kl = desired_kl
                self.metrics = [self.loss, self.average_kl_divergence]

                self.flat_params_op = get_flat_params(self.trainable_variables)
                self.loss_flat_gradients_op = get_flat_gradients(self.loss, self.trainable_variables)
                self.kl_flat_gradients_op = get_flat_gradients(self.average_kl_divergence, self.trainable_variables)

                self.vec = tf.placeholder(tf.float32, [None])
                self.fisher_product_op = self.get_fisher_product_op()

                self.new_params = tf.placeholder(tf.float32, [None])
                self.params_assign_op = assign_network_params_op(self.new_params, self.trainable_variables, self.trainable_variables_shapes)

            if(summary):
                self._create_summaries()

    def _create_summaries(self):
        with tf.name_scope('Summaries'):
            with tf.name_scope('Loss'):
                self.each_experience_loss_summary = tf.summary.histogram('each_experience_loss', self.each_experience_loss)
                self.average_experience_loss_summary = tf.summary.scalar('average_experience_loss', self.average_experience_loss)
                self.explained_variance_before_summary = tf.summary.scalar('explained_variance_before', self.explained_variance_before)
                self.explained_variance_after_summary = tf.summary.scalar('explained_variance_after', self.explained_variance_after)
            with tf.name_scope('Outputs'):
                self.hidden_layer1_summary = tf.summary.histogram('hidden_layer1', self.hidden_layer1)
                self.hidden_layer2_summary = tf.summary.histogram('hidden_layer2', self.hidden_layer2)
                self.hidden_layer3_summary = tf.summary.histogram('hidden_layer3', self.hidden_layer3)
                self.output_layer_summary = tf.summary.histogram('value_estimate', self.value_estimate)
            with tf.name_scope('Weights'):
                self.weights_hidden_layer1_summary = tf.summary.histogram('hidden_layer1', self.weights['h1'])
                self.weights_hidden_layer2_summary = tf.summary.histogram('hidden_layer2', self.weights['h2'])
                self.weights_hidden_layer3_summary = tf.summary.histogram('hidden_layer3', self.weights['h3'])
                self.weights_output_layer_summary = tf.summary.histogram('output_layer', self.weights['out'])
            with tf.name_scope('Biases'):
                self.biases_hidden_layer1_summary = tf.summary.histogram('hidden_layer1', self.biases['b1'])
                self.biases_hidden_layer2_summary = tf.summary.histogram('hidden_layer2', self.biases['b2'])
                self.biases_hidden_layer3_summary = tf.summary.histogram('hidden_layer3', self.biases['b3'])
                self.biases_output_layer_summary = tf.summary.histogram('output_layer', self.biases['out'])
            self.summary_op = tf.summary.merge([self.average_experience_loss_summary, self.each_experience_loss_summary, self.explained_variance_before_summary, self.explained_variance_after_summary, self.hidden_layer1_summary, self.hidden_layer2_summary, self.hidden_layer3_summary, self.output_layer_summary, self.weights_hidden_layer1_summary, self.weights_hidden_layer2_summary, self.weights_hidden_layer3_summary, self.weights_output_layer_summary, self.biases_hidden_layer1_summary, self.biases_hidden_layer2_summary, self.biases_hidden_layer3_summary, self.biases_output_layer_summary])


    def update_weights(self, states, targets):
        self.train_iteration += 1
        self.feed_dict = { self.state: states, self.target: targets  }
        loss, loss_gradients, old_predicted_targets, old_params = self.sess.run([self.loss, self.loss_flat_gradients_op, self.value_estimate, self.flat_params_op], self.feed_dict)
        self.feed_dict[self.old_predicted_targets] = old_predicted_targets
        self.feed_dict[self.variance] = loss
        step_direction = do_conjugate_gradient(self.get_fisher_product, loss_gradients)
        fisher_loss_product = self.getFisherProduct(step_direction)
        full_step_size = np.sqrt((2*self.desired_kl)/(np.dot(step_direction, fisher_loss_product)))
        full_step = full_step_size*step_direction
        #new_params = old_params + full_step
        new_params, status = do_line_search(self.get_metrics, old_params, full_step, self.desired_kl)
        if(status == False):
            self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return old_predicted_targets

    def predict_value(self, state):
        return self.sess.run(self.value_estimate, feed_dict = { self.state: state })

    def get_summary(self, states, targets, old_predicted_targets):
        feed_dict = { self.state: states, self.target: targets, self.old_predicted_targets: old_predicted_targets  }
        return self.sess.run(self.summary_op, feed_dict)

    def get_weights(self):
        return self.sess.run([self.weights, self.biases])

    def get_metrics(self, new_params):
        self.sess.run(self.params_assign_op, feed_dict = {self.new_params: new_params})
        return self.sess.run(self.metrics, self.feed_dict)
