import sys
import torch
import numpy as np
import pdb
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from encoder_model import SENTEncoder, RNNEncoder
from summ_model import CNN_z, FFN_z, FFN_pairwise_z, RNNV_z

#torch.manual_seed(0)
#np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class RNNVAE(nn.Module):

    def __init__(self, z_dim=20, h_dim=100, nwords=5000, embedding_dim=300, encoder="rnn", framework="rnnv",
            rnngate="lstm", device='cpu', z_combine="concat", delta_weight=1,scale_pzvar=1,
            freeze_weights=True, z_summary="average"):
        super(RNNVAE,self).__init__()
        self.embedding = nn.Embedding(nwords, embedding_dim, padding_idx=0)
        self.nwords = nwords
        self.z_summary = z_summary

        if encoder=="bert" or encoder=="infersent":
            # BERT always has 768 for h-dim and InferSent always has 4096
            self.encoder = SENTEncoder(z_dim=z_dim, h_dim=h_dim, encoder=encoder)
        elif encoder == "glove":
            self.encoder = RNNEncoder(z_dim=z_dim, h_dim=h_dim, rnngate=rnngate)
        else:
            sys.exit("encoder must be glove, bert, or infersent")


        if z_summary=="cnn":
            self.summariser_z = CNN_z(z_dim)
        elif z_summary=="ffn":
            self.summariser_z = FFN_z(z_dim, h_dim)
        elif z_summary=="ffn_pairwise":
            self.ffn_pairwise_z = FFN_pairwise_z(z_dim, h_dim)
            self.summariser_z = CNN_z(z_dim)
        elif z_summary=="rnn":
            self.summariser_z = RNNV_z(z_dim, variational=False)
        elif z_summary=="rnnv":
            self.summariser_z = RNNV_z(z_dim, variational=True)
        else:
            sys.exit("No summariser..!")

        # self.rnn_z


        self.encoder_name = encoder
        self.decoder = RNNDecoder(z_dim=z_dim, h_dim=h_dim, nwords=nwords, rnngate=rnngate, device=device)

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.device = device
        self.rnngate = rnngate
        self.framework = framework
        self.scale_pzvar = scale_pzvar
        self.cossim = nn.CosineSimilarity()

        self.delta_nn = DeltaPredictor(z_dim=z_dim, h_dim=h_dim, z_combine=z_combine)

        self.discrim_z = Discrim_z(h_dim, z_dim)
        #self.mse = nn.MSELoss()
        #self.mse = nn.HuberLoss()
        self.freeze = freeze_weights
        self.z_summary = z_summary


    # Try bilstm?
    def load_embeddings(self, fn=""):
        if len(fn)==0:
            self.embedding = nn.Embedding(self.nwords, 300, padding_idx=0)
        else:
            weights = torch.FloatTensor(np.loadtxt(fn))
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=self.freeze) #, padding_idx=0)

    def sample_z_reparam(self, q_mu, q_logvar):
        eps = torch.randn_like(q_logvar)
        z = q_mu + torch.exp(q_logvar*0.5) * eps
        return z.cuda()

    def encode_x(self, DP):
        if self.encoder_name == "bert" or self.encoder_name=="infersent":
            q_mu, q_logvar = self.encoder(DP[f'{self.encoder_name}_embed'])

        elif self.encoder_name == "glove":
            embed = self.embedding(DP['xx']).squeeze(0)
            q_mu, q_logvar = self.encoder(embed, DP['x_lens'])

        return q_mu, q_logvar


    def forward(self, DP):

        q_mu, q_logvar = self.encode_x(DP)

        if self.framework=="rnnv":
            zs = self.sample_z_reparam(q_mu, q_logvar)

        elif self.framework=="rnn":
            zs = q_mu

        eos_embed_seq = self.embedding(DP['ey']).squeeze(0)
        x_recon = self.decoder(DP['y_lens'], eos_embed_seq, zs)

        return x_recon, q_mu, q_logvar, zs

    def summarise_z(self, zs, OP_zs=None):
        #z_avg = torch.mean(z.squeeze(0), dim=0)
        # we kind either
        # 1. convolutions
        # 2. simple average
        # 3. weighted average, with weights learned from a nn
        if self.z_summary=="simple_average":
            z_avg = torch.mean(zs.squeeze(0), dim=0)
        else:
            z_avg = self.summariser_z(zs).squeeze(0)

        return z_avg


    def recon_loss(self, y, y_lens, x_recon, check=False):
        batch_ce_loss = 0.0
        y = y.squeeze(0)
        # y.size() = [nsentences, maxnwords]
        for i in range(y.size(0)):
            #ce_loss = F.cross_entropy(x_recon[i], y[i], reduction="sum", ignore_index=0)
            #ce_loss = ce_loss/y_lens[i].item()
            ce_loss = F.cross_entropy(x_recon[i], y[i], reduction="mean", ignore_index=0)
            batch_ce_loss += ce_loss

        batch_ce_loss = batch_ce_loss/y.size(0)
        return batch_ce_loss

    def temp_loss(self, q_mu, q_logvar):

        c = 1/self.scale_pzvar

        if q_mu.shape[1] > 3:
        #    larger_d_mu = (q_mu[0] - q_mu[3])**2 # we take the inner product
        #    smaller_d_mu = (q_mu[0] - q_mu[2])
        #    larger_kld = -0.5 * torch.sum( 1 + q_logvar - c*(q_logvar.exp()) - c*(larger_d_mu.pow(2)))
        #    smaller_kld = -0.5 * torch.sum( 1 + q_logvar - c*(q_logvar.exp()) - c*(d_mu.pow(2)))
            threshold = 0.01
            larger = [self.mse(q_mu[0,i,:], q_mu[0,i+2,:]) for i in range(q_mu.shape[1]-2)]
            smaller = [self.mse(q_mu[0,i,:], q_mu[0,i+1,:]) for i in range(q_mu.shape[1]-2)]

            larger = torch.stack(larger)
            smaller = torch.stack(smaller)

            losses = self._clamp_loss(smaller, larger, threshold)
            return losses

        else:
            return 0

        # threshold loss between kld1 and kld2


    def kld_loss(self, p_mu, q_mu, q_logvar):
        
        log01=np.log(self.scale_pzvar)
        c = 1/self.scale_pzvar
        d_mu = (p_mu - q_mu) # we take the inner product

        kld = -0.5 * torch.sum(-log01 + 1 + q_logvar - c*(q_logvar.exp()) - c*(d_mu.pow(2)))
 
#        kld = -0.5 * torch.sum(1 + q_logvar - q_mu.pow(2) - q_logvar.exp())
        #kld = kld

        return kld

    def loss_fnUE(self, y, x_recon, q_mu, q_logvar, embed):

        # scale_down_var = 0.1
        scale2=1/self.scale_pzvar

        log01 = torch.log(torch.ones_like(q_logvar)*self.scale_pzvar)
        identity_m = torch.eye(512).cuda()

        eqmu = (embed - q_mu)
        eqmuT = eqmu.transpose(1, 2)
        eqmuM = torch.matmul(torch.matmul(eqmu, identity_m*scale2), eqmuT)
        kld = -0.5 * (torch.sum(-log01 + 1 + q_logvar - scale2*q_logvar.exp()) - torch.sum(eqmuM))
        kld = kld/y.size(0)

        return batch_ce_loss, kld

    def contrast_loss(self, CO_pos_zsum, CO_neg_zsum, threshold):
        OH_contrast_dist = self.mse(CO_pos_zsum, CO_neg_zsum) 
        loss = torch.clamp((threshold - OH_contrast_dist), min=0)
        return loss

    def triplet_loss(self, OH_z, CO_pos_z, CO_neg_z, CO_irr_z, neg_out, threshold, hyp=1, weighted=0):
        # triplet loss
        # CHECK: how many elements in z
            #self.mse(OH_z, CO_pos_z) - self.mse(OH_z, CO_neg_z) > threshold
            # we expect the negative example to be closer rather than further!
        if weighted == 0:
            neg_out = None

        #OH_pos_dist = [self.mse(OH_z, CO_pos) for CO_pos in CO_pos_z]
        #OH_neg_dist = [self.mse(OH_z, CO_neg) for CO_neg in CO_neg_z]
        #OH_irr_dist = [self.mse(OH_z, CO_irr) for CO_irr in CO_irr_z]
        OH_z = OH_z.unsqueeze(0)

        OH_pos_dist = 1 - self.cossim(OH_z, CO_pos_z)
        OH_neg_dist = 1 - self.cossim(OH_z, CO_neg_z)
        OH_irr_dist = 1 - self.cossim(OH_z, CO_irr_z)

        #OH_pos_dist1 = [torch.cdist(OH_z, CO_pos.unsqueeze(0)).squeeze() for CO_pos in CO_pos_z]
        #OH_neg_dist2 = [torch.cdist(OH_z, CO_neg.unsqueeze(0)).squeeze() for CO_neg in CO_neg_z]
        #OH_irr_dist3 = [torch.cdist(OH_z, CO_irr.unsqueeze(0)).squeeze() for CO_irr in CO_irr_z]

        #import pdb; pdb.set_trace() 
        #OH_pos_dist = torch.stack(OH_pos_dist)
        #OH_neg_dist = torch.stack(OH_neg_dist)
        #OH_irr_dist = torch.stack(OH_irr_dist)
        
        if hyp == 0:
            return 0

        if hyp == 1:
            # positive examples are further from OH
            larger = OH_pos_dist
            smaller = OH_neg_dist

        elif hyp == 2:
             # negative examples are further from OH
            smaller = OH_pos_dist
            larger = OH_neg_dist

        elif hyp == 3:
            # irr examples are furthest from both OH_irr and OH_neg
            smaller = torch.cat((OH_pos_dist, OH_neg_dist))
            larger = OH_irr_dist

        elif hyp == 4:
            smallest = OH_neg_dist
            smaller = OH_pos_dist
            larger = OH_irr_dist

        elif hyp == 5:
            # write a hypothesis that says OH_irr_dist is always the smallest
            smallest = OH_pos_dist
            smaller = OH_neg_dist
            larger = OH_irr_dist


        if hyp in [1,2,3]:
            losses = self._clamp_loss(smaller, larger, threshold, neg_out)

        if hyp == 4 or hyp == 5:
            # this has 2 loss functions
            losses = self._clamp_loss(smaller, larger, threshold)
            losses += self._clamp_loss(smallest, smaller, threshold, neg_out)
        

        return losses

    def _clamp_loss(self, smaller, larger, threshold, neg_out=None):
        #loss = smaller - larger + threshold
#        if neg_out is not None:
#            neg_out = neg_out.detach().squeeze()
#            print(neg_out)
#            if len(neg_out.size())==0:
#                pass
#            else:
#                if smaller.shape[0] == neg_out.shape[0]:
#                    smaller = smaller[torch.where(neg_out>0.5)[0]]
#                else:
#                    larger = larger[torch.where(neg_out>0.5)[0]]


        x = smaller.repeat(larger.shape[0], 1) \
        - larger.repeat(smaller.shape[0], 1).transpose(0,1) + threshold
        
        # equivalent to max(x, 0), because we clamp the min value at 0
        losses = torch.clamp(x, min=0)
        #if neg_out is not None:
           # neg_out = -torch.log(neg_out)
        #    neg_out = 1-neg_out
           # losses = torch.exp(losses * neg_out.detach().squeeze())
        #    losses = torch.exp(losses * (-torch.log(neg_out.detach().squeeze())))
            #losses = losses[torch.where(neg_out>0.5)[0]]
            #neg_out = neg_out[torch.where(neg_out<0.5)[0]]

            #losses = losses * neg_out.detach().squeeze()
    
        if len(torch.nonzero(losses))>0:
        #    losses = losses.sum()/len(torch.nonzero(losses)) # average across nonzero only
            losses = losses.mean()
        else:
            losses = 0
        return losses

    # if threshold neg, no triplet loss


class DeltaPredictor(nn.Module):
    def __init__(self, z_dim, h_dim, z_combine="concat", device="cpu"):
        super(DeltaPredictor, self).__init__()

        self.z_combine = z_combine
        self.device = device

        if z_combine=="concat":
            #self.fc1 = nn.Linear(z_dim*2+1, h_dim)
            self.fc1 = nn.Linear(z_dim*2, h_dim)

        elif z_combine=="diff":
            #self.fc1 = nn.Linear(z_dim+1, h_dim)
            self.fc1 = nn.Linear(z_dim, h_dim)

        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, 1)

    def forward(self, z1, z2, length=None):


        z1 = z1.cuda()
        z2 = z2.cuda()

        if self.z_combine == "concat":
            zz = torch.cat((z1, z2), 1)

        elif self.z_combine == "diff":
            zz = torch.sub(z2, z1)
        
        h1 = F.relu(self.fc1(zz))
        h2 = F.relu(self.fc2(h1))

        out = torch.sigmoid(self.fc3(h2))
        # convert to BCEWithLogits
        # convert everything to log form 
        #out = self.fc3(h2)
        return out

    def delta_loss(self, out, delta=None):
        delta = torch.Tensor([delta]).cuda().repeat(len(out))
        if out.size(0)==1:
            out = out.squeeze(0)
        else:
            out = out.squeeze()
        
        return F.binary_cross_entropy(out, delta, reduction="mean")

    def margin_loss(self, pos_pred, neg_pred, margin=0.5):
        # ranking loss 
        losses = neg_pred.squeeze() - pos_pred + margin
        loss = torch.clamp(losses, min=0).mean()

        return loss

    def pairwise_predict(self, y_pos, y_neg):
        eq = 0
        score = 0

        if y_pos>y_neg:
            score += 2
        if y_pos == y_neg:
            eq += 2
            
        return score, eq




class RNNDecoder(nn.Module):
    def __init__(self, z_dim=20,
                        h_dim=100,
                        embedding_dim=300,
                        n_layers=1,
                        nwords=5000,
                        rnngate="lstm",
                        device='cpu'):

        super(RNNDecoder, self).__init__()
        self.fc_z_h = nn.Linear(z_dim, h_dim)
        self.fc_z_c = nn.Linear(z_dim, h_dim)
        self.h_dim = h_dim
        self.nn = getattr(nn, rnngate.upper())(embedding_dim, h_dim, n_layers,
                batch_first=True)
        self.fc_out = nn.Linear(h_dim, nwords)
        self.rnngate = rnngate
        self.device = device

    def forward(self, y_length, embed, z):

        hidden = self.fc_z_h(z)
        ccell = torch.tanh(hidden)

        packed = pack_padded_sequence(embed, y_length, batch_first=True, enforce_sorted=False)
        if self.rnngate=="lstm":
        #    ccell = torch.randn(hidden.size()).cuda()
            outputs, _ = self.nn(packed, (hidden, ccell ))
        else:
            outputs, _ = self.nn(packed, hidden)

        outputs, _ = pad_packed_sequence(outputs, batch_first=True,
                total_length=max(y_length).item())

        outputs = self.fc_out(outputs)
        # outputs.shape = [n_sentences, longest_sent, nvocab_words]

       #outputs = F.logsoftmax(self.fc_out(outputs))
        return outputs

    def rollout_decode(self, input0, z, embedding, true_ix=[]):

        all_decoded = []
        all_probs = []

        if z.dim()==1:
            z = z.unsqueeze(0)

        z = z.cuda()
        hiddens = self.fc_z_h(z)

        for i in range(z.size(0)):
            decoded = []
            probs = []

            first_hidden = hiddens[i].view(1, 1, self.h_dim)
            ccell = torch.tanh(hiddens[i].view(1, 1, self.h_dim)).cuda()
            output0, (hidden, cell) = self.nn(input0.unsqueeze(0), (first_hidden, ccell))

            vocab_probs = F.softmax(self.fc_out(output0).squeeze(), dim=0)

            output_ix = torch.argmax(vocab_probs).item()
            decoded.append(output_ix)

            if len(true_ix)!=0:
                probs.append(vocab_probs[true_ix[0]])

            output = output_ix

            j=0
            while (len(decoded)<30):
                j+=1

                outputx = embedding(torch.LongTensor([output_ix]).cuda())
                output_h, (hidden, cell) = self.nn(outputx.unsqueeze(0), (hidden, cell))
                vocab_probs = F.softmax(self.fc_out(output_h).squeeze(), dim=0)
                output_ix = torch.argmax(vocab_probs).item()

                decoded.append(output_ix)
                if len(true_ix)!=0:
                    probs.append(vocab_probs[true_ix[j]])

            all_decoded.append(decoded)
            all_probs.append(probs)

        return all_decoded, all_probs

# AAE
class Discrim_z(nn.Module):
    def __init__(self, N, z_dim):
        super(Discrim_z, self).__init__()
        self.fc1 = nn.Linear(z_dim, N)
        self.fc2 = nn.Linear(N, N)
        self.fc3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


        #elif self.z_summary == "similarity" or self.z_summary=="dissimilarity":
            # attention based on cosine similarity and dissimilarity
            # we are dealing with OP, just use z_avg for now.
            #if debug:
            #    pdb.set_trace()
        #    if OP_zs is None:
        #        z_avg = torch.mean(zs.squeeze(0), dim=0)
        #    else:
                # to get summary of CO paragraph
                # calculate the similarity of each sentence, to the overall OP_z
                # weigh each sentence in CO based on similarity to OP_z
        #        OP_z = OP_zs.unsqueeze(0)
        #        zs = zs.squeeze(0)
                # take cosinesim, a.b/(|a||b|) 
                # exp(cossim)/Z
        #        sims = torch.mm(OP_z, torch.transpose(zs, 1, 0))/(self.z_dim**2)
        #        weights = torch.exp(sims)/ torch.exp(sims).sum()

        #        if self.z_summary=="dissimilarity":
        #            sims = 1 - sims # distance
        #            weights = sims/sims.sum()
                # take weighted average (based on smilarity)
         #       z_avg = torch.sum(torch.transpose(weights, 1, 0).repeat(1, zs.shape[1]) * zs, dim=0)


