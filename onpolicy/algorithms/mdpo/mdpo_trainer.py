import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.ppo_epoch = args.num_mini_batch # DEBUG_MK I thought this would have been 15. So that's not a problem
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = 0.1 # DEBUG_MK args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta
        self.num_env_steps = args.num_env_steps
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.kl_threshold = args.kl_threshold
        # self.ls_step = args.ls_step
        self.total_episodes = float(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        self.current_episode = 0

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

    def create_new_params(self, current_flat_params, t_k, gradients) -> torch.Tensor:
        """
        Creates new parameters based on the current flat parameters, time step t_k, and gradients.
        Args:
            current_flat_params (torch.Tensor): Current flat parameters.
            t_k (float): Step size.
            gradients (torch.Tensor): Gradients.
        Returns:
            torch.Tensor: New parameters.
        """
        interList = []
        for param,grad in zip(current_flat_params, gradients):
            temp = param * torch.exp(-t_k * grad)
            interList.append(temp)

        interList = torch.cat(interList)
        total = torch.sum(interList)

        new_params = []

        for param in interList:
            new_params.append(param/total)
        new_params = torch.cat(new_params)
        return new_params
    
    

    


    # new_actor, old_actor, obs, rnn_states, actions, masks, available_actions, active_masks
    # old_mu, new_mu, old_std, new_std, old_probs, new_probs
    def kl_divergence(self,  old_probs, new_probs, old_log_probs, new_log_probs):
        
        # old_mu = old_probs.mean()
        # new_mu = new_probs.mean()
        # old_std = torch.std(old_probs)
        # new_std = torch.std(new_probs)

        # print("old_mu:", old_mu)
        # print("new_mu:", new_mu)
        # print("old log probs:", old_log_probs)
        # print("new log probs", new_log_probs)
        # print("old_probs:", old_probs)
        # print("new_probs:", new_probs)
        

        calculated_new_probs = torch.exp(new_log_probs)
        # We want KL(new||old) not D( pi_old || pi_new )
        if isinstance(new_probs, torch.Tensor):  # Categorical distribution
            kl = torch.sum(calculated_new_probs * (new_log_probs - old_log_probs), dim=-1)
            # print("Categorical distribution")
        
        else:
            kl = torch.tensor(0.0, requires_grad=True, device=self.device)
        # else:  # Gaussian distribution
        #     old_logstd = torch.log(old_std)
        #     new_logstd = torch.log(new_std).requires_grad_()


            
        #     kl = torch.sum(
        #         old_logstd - new_logstd + 
        #         (new_std.pow(2) + (new_mu - old_mu).pow(2)) / (2 * old_std.pow(2)) - 0.5,
        #         dim=-1
        #     )
            
        
        return kl.mean()
        
    #
    # obs, rnn_states, actions, masks, available_actions, active_masks
    def compute_kl_gradient(self, old_mu, new_mu, old_std, new_std, old_probs, new_probs):


        # Compute KL divergence : new_policy, old_policy, obs, rnn_states, actions, masks, available_actions, active_masks
        kl = self.kl_divergence(old_mu, new_mu, old_std, new_std, old_probs, new_probs)

        # Compute gradient of KL with respect to new policy parameters
        kl_grad = torch.autograd.grad(kl.mean(), self.policy.actor.parameters(), create_graph=True,allow_unused=True) 
        # print("kl grad shape:", len(kl_grad))
        # print("KL value", kl)
        # print("KL grad:", kl_grad)      
        return kl, kl_grad
    
    def compute_first_term(self, obs, rnn_states, actions, masks, available_actions, active_masks, oldActorLogProbs, advantage):
        newActionLogProbs, dist_entropy , new_mu, new_std, newProbs =self.policy.actor.evaluate_actions(obs, rnn_states, actions, masks, available_actions, active_masks)
        
        
        ratio = torch.exp(newActionLogProbs - oldActorLogProbs)
        intermediate = -torch.sum(newActionLogProbs*advantage).mean() # Intermediate should be a scalar to perform differentiation

        logAdvGrad = torch.autograd.grad(intermediate, self.policy.actor.parameters(), create_graph=True)


        ratio_scalar = torch.sum(ratio, dim=-1, keepdim=True).mean() 
        firstTerm = tuple([ratio_scalar*x for x in logAdvGrad])

        return firstTerm



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

        # critic update

        
        value_loss = self.cal_val_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()
        
        # DEBUG_MK: self.value_loss_coef is hardcoded to 0.01
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
        old_params = self.flat_params(self.policy.actor)
        old_actor = R_Actor(self.policy.args, 
                            self.policy.obs_space,  
                            self.policy.act_space, 
                            self.device)
        self.update_model(old_actor, old_params)

        _, _, old_mu, old_std, old_probs = old_actor.evaluate_actions(obs_batch, rnn_states_batch, actions_batch, masks_batch, available_actions_batch, active_masks_batch) 
        oldLogProbs = action_log_probs

        actor_grad_norm = 0
        kl_final = 0

        sgd_steps:int = 10 # Number of steps for SGD
        for step in range(sgd_steps):

            # step_size:float = 1 - (step/10) # This is one method
            step_size:float = self.total_episodes/(self.total_episodes + 1 - self.current_episode) 
            
            
            """Manually gradient calculation"""
            new_action_log_probs, new_entropy, new_mu, new_std, new_probs = self.policy.actor.evaluate_actions(obs_batch, rnn_states_batch, actions_batch, masks_batch, available_actions_batch, active_masks_batch)

            # kl, kl_grad = self.compute_kl_gradient(old_mu, new_mu, old_std, new_std, old_probs, new_probs) 
                      
            # firstTerm = self.compute_first_term(obs_batch, rnn_states_batch, actions_batch, masks_batch, available_actions_batch, active_masks_batch, oldLogProbs, adv_targ) 

            # finalGradient = E[pi_new / pi_old * (nabla_theta pi_new)*Advantage] - 1/t_k * nabla_KL(pi_old || pi_new)
            # finalGradient = tuple([(firstTerm[i] - step_size*kl_grad[i])*self.value_loss_coef for i in range(len(firstTerm))])

            # manualPsiGrad = self.flat_grad(finalGradient)
            # print(finalGradient)

            # print("Manual grad: ", finalGradient.mean()) 

            """ Automatic gradient calculation """
            imp_weights = torch.exp(new_action_log_probs - old_action_log_probs_batch)
            adv_term = imp_weights * adv_targ
            kl = self.kl_divergence( old_probs, new_probs, old_action_log_probs_batch, new_action_log_probs)

            # adv_term = torch.sum(surr1, dim=-1, keepdim=True)


            if step == 0:
                psi_val_inter = -torch.mean(adv_term)
            else:
                psi_val_inter = -(torch.mean(adv_term) - kl*step_size) # psi = E[E[advantage] - KL(pi_old, pi_new)]
            
            psi_val = (psi_val_inter * self.value_loss_coef)
            
            self.policy.actor_optimizer.zero_grad()
            
            psi_val.backward(retain_graph=True)

            actor_grad_norm= nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            # actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

            self.policy.actor_optimizer.step()

            # psi_grad = torch.autograd.grad(psi_val, self.policy.actor.parameters(), allow_unused=True) # built in gradient calculation

            # autoPsiGrad = self.flat_grad(psi_grad)
            

            
            
            """ Manually updating actor params"""

            # params = self.flat_params(self.policy.actor)
            # new_params = params - manualPsiGrad

            # self.update_model(self.policy.actor, new_params)
            # actor_grad_norm = torch.norm(manualPsiGrad).item()
            
            # print("Iteration: ", step)

            # print("Average gradient: ", manualPsiGrad.mean())
            # print("Step size: ", step_size)
            # print("Actor_grad_norm: ", actor_grad_norm)
            # print("KL divergence: ", kl.mean())
            # print("Psi value: ", psi_val) # psi is nan


            if step == sgd_steps - 1:
                kl_final = kl.mean()
                

            # actor_grad_norm= get_gard_norm(self.policy.actor.parameters())
            # self.policy.actor_optimizer.step() # Removed this to manually update the actor params

        
        # print("Actor_grad_norm: ", actor_grad_norm)
        # print(psi_val.mean())
        # exit()

        # return value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, ratio
        return value_loss, critic_grad_norm, actor_grad_norm, kl_final, dist_entropy, psi_val


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
        train_info['episode_nume'] = 0
        
        data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

 
        for sample in data_generator:
            # DEBUG_MK: , imp_weights <-- Problems. When adding back, change in mdpo_update too
            value_loss, critic_grad_norm,actor_grad_norm ,kl , dist_entropy, psi_val\
                = self.mdpo_update(sample, update_actor)


            

            train_info['value_loss'] += value_loss.item()
            train_info['kl'] += kl.item()
            # train_info['loss_improve'] += loss_improve.item()
            # train_info['expected_improve'] += expected_improve
            train_info['dist_entropy'] += dist_entropy.item()
            train_info['critic_grad_norm'] += critic_grad_norm
            train_info['actor_grad_norm'] += actor_grad_norm
            # train_info['ratio'] += imp_weights.mean()
            train_info['psi'] += psi_val.item()
            train_info['episode_nume'] = self.current_episode
                

                
            
        num_updates = 10 *  self.num_mini_batch
        
        self.current_episode += 1
        
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info
    
    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
