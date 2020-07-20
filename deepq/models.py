import tensorflow as tf
import tensorflow.contrib.layers as layers

def _mlp_branching(hiddens_common, hiddens_actions, num_action_branches,
                   struct, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        print("STRUCTURE:")
        print(struct)
        out = inpt

        # Create the shared network module (unless independent)
        with tf.variable_scope('common_net'):
            for hidden in hiddens_common:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)

        # Create the action branches
        with tf.variable_scope('action_value'):
            total_action_scores = []
            if struct is None:
                #Assume it's parallel
                for action_stream in range(num_action_branches):
                    action_out = out
                    for hidden in hiddens_actions:
                        action_out = layers.fully_connected(action_out, num_outputs=hidden,
                                                            activation_fn=tf.nn.relu)
                    action_scores = layers.fully_connected(action_out,
                                                           num_outputs=num_actions // num_action_branches,
                                                           activation_fn=None)
                    total_action_scores.append(action_scores)
            else:
                total_action_scores = [None] * 17
                mid_layers = [None] * 17
                for layer in struct:
                    for module in layer:
                        action_out = out
                        for hidden in hiddens_actions:
                            action_out = layers.fully_connected(action_out, num_outputs=hidden,
                                                                activation_fn=tf.nn.relu)
                            mid_layers[module] = action_out
                        action_scores = layers.fully_connected(action_out,
                                                               num_outputs=num_actions // num_action_branches,
                                                               activation_fn=None)
                        total_action_scores[module] = action_scores
                    for module_i, module in enumerate(layer):
                        out = tf.concat([out, tf.stop_gradient(tf.nn.softmax(
                                total_action_scores[module] - tf.reduce_min(total_action_scores[module], 1,
                                                                            True)))], 1)
    return total_action_scores

def _mlp_noisy_branching(hiddens_common, hiddens_actions, num_action_branches,
                   struct, inpt, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        print("STRUCTURE:")
        print(struct)
        out = inpt

        # Create the shared network module (unless independent)
        with tf.variable_scope('common_net'):
            for hidden in hiddens_common:
                out = layers.fully_connected(out, num_outputs=hidden, activation_fn=tf.nn.relu)

        # Create the action branches
        with tf.variable_scope('action_value'):

            total_action_scores = []
            if struct is None:
                ### Assume it's parallel
                for action_stream in range(num_action_branches):
                    action_out = out
                    for hidden in hiddens_actions:
                        action_out = layers.fully_connected(action_out, num_outputs=hidden,
                                                            activation_fn=tf.nn.relu)
                    action_scores = layers.fully_connected(action_out,
                                                           num_outputs=num_actions // num_action_branches,
                                                           activation_fn=None)
                    total_action_scores.append(action_scores)
            else:

                total_action_scores = [None] * 17
                mid_layers = [None] * 17
                for layer in struct:
                    for module in layer:
                        action_out = out
                        for hidden in hiddens_actions:
                            action_out = noisy_dense(action_out,name= str(module)+str(hidden), size=hidden,
                                                                activation_fn=tf.nn.relu)
                            # action_out = tf.nn.dropout(action_out, keep_prob=0.7)
                            mid_layers[module] = action_out
                        action_scores = noisy_dense(action_out,name= str(module)+"out",
                                                    size=num_actions // num_action_branches,
                                                               activation_fn=None)
                        total_action_scores[module] = action_scores
                    for module_i, module in enumerate(layer):
                        out = tf.concat([out, tf.stop_gradient(tf.nn.softmax(
                                total_action_scores[module] - tf.reduce_min(total_action_scores[module], 1,
                                                                            True)))], 1)
    return total_action_scores




def mlp_branching(hiddens_common=[], hiddens_actions=[], num_action_branches=None,
                  struct=[[0, 1, 2]]):
    """This model takes as input an observation and returns values of all sub-actions -- either by
    combining the state value and the sub-action advantages (i.e. dueling), or directly the Q-values.

    Parameters
    ----------
    hiddens_common: [int]
        list of sizes of hidden layers in the shared network module --
        if this is an empty list, then the learners across the branches
        are considered 'independent'

    hiddens_actions: [int]
        list of sizes of hidden layers in the action-value/advantage branches --
        currently assumed the same across all such branches

    num_action_branches: int
        number of action branches (= num_action_dims in current implementation)

    dueling: bool
        if using dueling, then the network structure becomes similar to that of
        dueling (i.e. Q = f(V,A)), but with N advantage branches as opposed to only one,
        and if not dueling, then there will be N branches of Q-values



    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp_branching(hiddens_common, hiddens_actions,
                                                  num_action_branches,
                                                  struct, *args, **kwargs)


def mlp_noisy_branching(hiddens_common=[], hiddens_actions=[], num_action_branches=None,
                  struct=[[0, 1, 2]]):
    """This model takes as input an observation and returns values of all sub-actions -- either by
    combining the state value and the sub-action advantages (i.e. dueling), or directly the Q-values.

    Parameters
    ----------
    hiddens_common: [int]
        list of sizes of hidden layers in the shared network module --
        if this is an empty list, then the learners across the branches
        are considered 'independent'

    hiddens_actions: [int]
        list of sizes of hidden layers in the action-value/advantage branches --
        currently assumed the same across all such branches

    num_action_branches: int
        number of action branches (= num_action_dims in current implementation)

    dueling: bool
        if using dueling, then the network structure becomes similar to that of
        dueling (i.e. Q = f(V,A)), but with N advantage branches as opposed to only one,
        and if not dueling, then there will be N branches of Q-values

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp_noisy_branching(hiddens_common, hiddens_actions, num_action_branches,
                                                  struct,  *args, **kwargs)
