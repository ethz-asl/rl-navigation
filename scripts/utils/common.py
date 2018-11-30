import tensorflow as tf
import numpy as np

def get_variance(x):
    _, var = tf.nn.moments(x, axes=[0])
    return var

def get_generalized_advantages(td_errors, discount_factor, lamda, terminal_state_estimated_value):
    td_errors[-1] -= lamda*terminal_state_estimated_value #Account for the terminal state
    delta_factor = discount_factor*lamda
    generalized_advantages = [0]*len(td_errors)
    a = 0
    for t in reversed(range(len(td_errors))):
        a = td_errors[t] + delta_factor * a
        generalized_advantages[t] = a
    return np.asarray(generalized_advantages)

def do_conjugate_gradient(f_Ax, g, iterations = 20, tolerance = 1e-10):
    x_i = np.zeros_like(g)
    r_i = np.copy(g)
    b_i = np.copy(r_i)
    r_i_dot_r_i = np.dot(r_i, r_i)
    for i in range(iterations):
        A_b_i = f_Ax(b_i)
        alpha_i = np.squeeze(r_i_dot_r_i/np.dot(b_i, A_b_i))
        x_i = x_i + alpha_i*b_i
        r_i_new = r_i - alpha_i*A_b_i
        r_i_dot_r_i_new = np.dot(r_i_new, r_i_new)
        beta_i = np.squeeze(r_i_dot_r_i_new/r_i_dot_r_i)
        r_i = r_i_new
        r_i_dot_r_i = r_i_dot_r_i_new
        if(r_i_dot_r_i < tolerance):
            break
        b_i = r_i + beta_i*b_i
    return x_i

def do_line_search(f, old_params, full_step, desired_kl_divergence, max_backtracks = 15):
    old_loss, old_kl_divergence = f(old_params)
    for (backtrack, step_frac) in enumerate(.8**np.arange(max_backtracks)):
        new_params = old_params - step_frac*full_step
        new_loss, new_kl_divergence = f(new_params)
        if new_loss < old_loss and new_kl_divergence < desired_kl_divergence:
            return new_params, True
    return old_params, False

def do_line_search_CPO(f, old_params, full_step, desired_kl_divergence, linear_constraint_threshold, check_loss = True, max_backtracks = 15):
    old_loss, old_kl_divergence, old_safety_loss = f(old_params)
    for (backtrack, step_frac) in enumerate(.8**np.arange(max_backtracks)):
        new_params = old_params - step_frac*full_step
        new_loss, new_kl_divergence, new_safety_loss = f(new_params)
        if ((not(check_loss) or (new_loss < old_loss)) and (new_kl_divergence < desired_kl_divergence) and (new_safety_loss <= linear_constraint_threshold)):
            return new_params, True
    return old_params, False

def get_2D_slice(mat, indices):
    mat_shape = tf.shape(mat)
    n_rows, n_cols = mat_shape[0], mat_shape[1]
    ind_mul = tf.range(n_rows)
    mat_flat = tf.reshape(mat, [-1])
    return tf.gather(mat_flat, ind_mul*n_cols + indices)

def get_flat_gradients(loss, var_list):
    return tf.concat(list(map(lambda x: tf.reshape(x,[-1]), tf.gradients(loss, var_list))), axis = 0)

def get_flat_params(var_list):
    return tf.concat(list(map(lambda x: tf.reshape(x,[-1]), var_list)), axis = 0)

def assign_network_params_op(params, var_list, var_shapes):
    assigns = []
    start = 0
    for shape, var in zip(var_shapes, var_list):
        var_size = np.prod(shape)
        assigns.append(tf.assign(var,tf.reshape(params[start:start+var_size], shape)))
        start += var_size
    return tf.group(*assigns)
