import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.algorithms.utils.popart_hatrpo import PopArt
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

class MDPO():

    """ MDPO Policy class"""
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.num_mini_batch = args.num_mini_batch
        self.ppo_epoch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = 1 # DEBUG_MK args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self.kl_threshold = args.kl_threshold
        self.ls_step = args.ls_step
        self.accept_ratio = args.accept_ratio

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None
    
    def cal_val_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart or self._use_valuenorm:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped # This is not needed. TODO: Get rid of this
            error_original = return_batch - values # This is fine

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss
        
    def flat_grad(self, grads):
        """
        Flattens the gradients into a 1-dimensional tensor.

        Args:
            grads (list): A list of gradients.

        Returns:
            torch.Tensor: A 1-dimensional tensor containing the flattened gradients.
        """
        grad_flatten = []
        for grad in grads:
            if grad is None:
                continue
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            if hessian is None:
                continue
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten

    def flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length
    
    def kl_approx(self, q, p):
        r = torch.exp(p - q)
        kl = r - 1 - p + q
        return kl
    
    def kl_divergence(self, obs, rnn_states, action, masks, available_actions, active_masks, new_actor, old_actor):
        """
        Calculates the Kullback-Leibler (KL) divergence between the old policy and the new policy.

        Parameters:
        - obs (Tensor): The observation tensor.
        - rnn_states (Tensor): The recurrent neural network (RNN) states tensor.
        - action (Tensor): The action tensor.
        - masks (Tensor): The mask tensor.
        - available_actions (Tensor): The tensor containing available actions.
        - active_masks (Tensor): The tensor containing active masks.
        - new_actor (Actor): The new actor network.
        - old_actor (Actor): The old actor network.

        Returns:
        - kl (Tensor): The KL divergence tensor.

        Note:
        - The KL divergence is a measure of how one probability distribution differs from another.
        - It is used to compare the old policy (pi_old) with the new policy (pi_new).
        - The KL divergence is not a symmetric metric, so be careful when calculating it.
        """
        
        _, _, mu, std, probs = new_actor.evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks)
        _, _, mu_old, std_old, probs_old = old_actor.evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks)
        if mu.grad_fn==None:
            probs_old=probs_old.detach()
            kl = self.kl_approx(probs_old,probs)
        else:
            logstd = torch.log(std)
            mu_old = mu_old.detach()
            std_old = std_old.detach()
            logstd_old = torch.log(std_old)
            # kl divergence between old policy and new policy : D( pi_old || pi_new )
            # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
            # be careful of calculating KL-divergence. It is not symmetric metric
            kl =  logstd - logstd_old  + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
        
        if len(kl.shape)>1:
            kl=kl.sum(1, keepdim=True)
        return kl
    
    def mdpo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks using MDPO
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        # Checking if the object is a numpy array and converting it to a tensor
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        #  DEBUG_MK: factor_batch # I don't know where this is used. We'll get to this later
        """
        adv_targ := Advantage value?? I think 
        old_action_log_probs_batch := Old action policy 
        
        """

        values, action_log_probs, dist_entropy, action_mu, action_std, _  = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # DEBUG_MK: Reached here
        # imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        # critic update
        value_loss = self.cal_val_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        
        # DEBUG_MK: self.value_loss_coef is hardcoded to 1
        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        """ 
        Actor update steps: 
        1. Initialise an "old" actor with current actor params
        2. for loop for updating?
        """
        #Reached here
        old_params = self.flat_params(self.policy.actor)
        old_actor = R_Actor(self.policy.args, 
                            self.policy.obs_space,  
                            self.policy.act_space, 
                            self.device)
        self.update_model(old_actor, old_params)

        actor_grad_norm = 0

        sgd_steps:int = 10 # Number of steps for SGD
        for step in range(sgd_steps):

            step_size:float = 1 - (step/10) # This is one method
            kl = self.kl_divergence(obs_batch, 
                               rnn_states_batch, 
                               actions_batch, 
                               masks_batch, 
                               available_actions_batch, 
                               active_masks_batch,
                               new_actor=self.policy.actor,
                               old_actor=old_actor)
            kl_expected = kl.mean()   


            # psi = E[E[advantage] - KL(pi_old, pi_new)]
            print("Calculating psi... Might crash here")
            psi_val = (adv_targ.mean() - kl_expected*step_size).mean() # This should already be a scalar
            
            self.policy.actor_optimizer.zero_grad()
            # method 1
            
            psi_val.backward(inputs=self.policy.actor.parameters()) # DEBUG_MK: Not sure if this works at all
            
            #DEBUG_MK: Why is this zero
            actor_grad_norm, none_counter = get_gard_norm(self.policy.actor.parameters())

            self.policy.actor_optimizer.step()

            # method 2 
            # psi_grad = torch.autograd.grad(psi_val, self.policy.actor.parameters(), allow_unused=True)
            # upadte
            

        # return value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, ratio
        return value_loss, critic_grad_norm, actor_grad_norm, kl, dist_entropy, psi_val, none_counter


    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD. Train function is the same for all algorithms
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        # Not adding the popart flag. I don't know what that does. So lmao
        advantages = buffer.returns[:-1] - buffer.value_preds[:-1] # Q Value function - Value Function
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        train_info = {}
        train_info['value_loss'] = 0
        train_info['kl'] = 0
        train_info['dist_entropy'] = 0
        # train_info['loss_improve'] = 0
        # train_info['expected_improve'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['actor_grad_norm'] = 0
        # train_info['ratio'] = 0
        train_info['psi'] = 0
        train_info['none_counter'] = 0 # DEBUG_MK: Counting how many times the gradient is None


        data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

        for epoch in range(self.ppo_epoch):    
            for sample in data_generator:
                # DEBUG_MK: , imp_weights <-- Problems. When adding back, change in mdpo_update too
                value_loss, critic_grad_norm,actor_grad_norm ,kl , dist_entropy, psi_val, none_counter \
                    = self.mdpo_update(sample, update_actor)

                print(type(psi_val))
                

                train_info['value_loss'] += value_loss.item()
                train_info['kl'] += kl.mean()
                # train_info['loss_improve'] += loss_improve.item()
                # train_info['expected_improve'] += expected_improve
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['actor_grad_norm'] += actor_grad_norm
                # train_info['ratio'] += imp_weights.mean()
                train_info['psi'] = psi_val.mean()
                train_info['none_counter'] = none_counter

                
            if epoch == 0 : print("Trained for 1 epoch successfully")
            num_updates = self.ppo_epoch*  self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info
    
    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
