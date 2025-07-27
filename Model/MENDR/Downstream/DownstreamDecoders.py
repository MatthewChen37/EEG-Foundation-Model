import torch
import torch.nn as nn
from Model.MENDR.mAtt.spd import SPDTangentSpace

class TUABFinetuneDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tangent = SPDTangentSpace(19)
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(nn.Linear(5*19*10, 5*19*10), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(5*19*10, 5*19*10))
        #self.seq = nn.Sequential(nn.Linear(1*1*10, 1*1*10), nn.LayerNorm(1*19*10), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*10, 1*1*10))
        #self.final_decoder = nn.Sequential(nn.GELU(), nn.Linear(5*5*11+5*19*1, 1))
        self.final_decoder = nn.Sequential(nn.GELU(), nn.Dropout(0.1), nn.Linear(5*19*10, 1))
        #self.final_decoder = nn.Sequential(nn.GELU(), nn.Linear(5*19*1, 1))
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        '''
        self.decoders = nn.ParameterDict({
            'delta': nn.Sequential(nn.Linear(5*19*96, 1*19*32), nn.LayerNorm(1*19*32), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*32, 1*19*1)), 
            'theta': nn.Sequential(nn.Linear(5*19*96, 1*19*32), nn.LayerNorm(1*19*32), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*32, 1*19*1)),
            'alpha': nn.Sequential(nn.Linear(5*19*192, 1*19*32), nn.LayerNorm(1*19*32), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*32, 1*19*1)),
            'beta': nn.Sequential(nn.Linear(5*19*384, 1*19*32), nn.LayerNorm(1*19*32), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*32, 1*19*1)),
            'gamma': nn.Sequential(nn.Linear(5*19*768, 1*19*32), nn.LayerNorm(1*19*32), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*32, 1*19*1)),
            #'high': nn.Sequential(nn.Linear(5*19*512, 1*19*512), nn.LayerNorm(1*19*512), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(1*19*512, 1*19*1), nn.LayerNorm(1*19*1)),
        })
        '''
        #self.final_decoder = nn.Sequential(nn.GELU(), nn.Linear(5*19*1, 1))

    '''
    def forward(self, encodings):
        feautrizations = dict()
        for band, decoder in self.decoders.items():
            feautrizations[band] = decoder(self.flatten(encodings[band]))
        x = torch.cat(list(feautrizations.values()), dim=-1)
        return self.final_decoder(x)
    '''
        
    def forward(self, x):
        batch_size = x.shape[0]
        num_patches = x.shape[1]
        embedding_dim = x.shape[2]
        x = self.tangent(x.view(batch_size*num_patches, embedding_dim, embedding_dim))
        #print(x.shape)
        #x = self.adaptive_pool(x)
        x = x.view(batch_size, num_patches, -1)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        #x = self.seq(x)
        #print(x.shape)
        '''
        feautrizations = dict()
        for band, decoder in self.decoders.items():
            feautrizations[band] = decoder(self.flatten(encodings[band]))
        feautrizations = torch.cat(list(feautrizations.values()), dim=-1)
        return self.final_decoder(feautrizations)
        #return self.final_decoder(torch.cat([x, feautrizations], dim=-1))
        '''
        x = self.seq(x)
        return self.final_decoder(x)

class TUEVFinetuneDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tangent = SPDTangentSpace(19)
        self.flatten = nn.Flatten()
        self.flattened = (19 * 20) // 2
        
        #self.combined_seq = nn.Sequential(nn.LayerNorm(5 * 100), nn.GELU(), nn.Linear(5*100, 100), nn.LayerNorm(100), nn.GELU())
        #self.final_lin = nn.Linear(100, 6)
        '''
        self.band_tangent_spaces = nn.ParameterDict({
            'delta': SPDTangentSpace(19),
            'theta': SPDTangentSpace(19),
            'alpha': SPDTangentSpace(19),
            'beta':  SPDTangentSpace(19),
            'gamma': SPDTangentSpace(19),
        })

        self.band_flatten = nn.ParameterDict({
            'delta': nn.Flatten(),
            'theta': nn.Flatten(),
            'alpha': nn.Flatten(),
            'beta':  nn.Flatten(),
            'gamma': nn.Flatten(),
        })

        self.band_dim = {
            'delta': 19,
            'theta': 19,
            'alpha': 19,
            'beta':  19,
            'gamma': 19,
        }

        self.band_decoders = nn.ParameterDict({
            'delta': nn.Sequential(nn.Linear(self.flattened, 30), nn.LayerNorm(30), nn.Dropout(p=0.2)),
            'theta': nn.Sequential(nn.Linear(self.flattened, 30), nn.LayerNorm(30), nn.Dropout(p=0.2)),
            'alpha': nn.Sequential(nn.Linear(self.flattened, 30), nn.LayerNorm(30), nn.Dropout(p=0.2)),
            'beta':  nn.Sequential(nn.Linear(self.flattened, 30), nn.LayerNorm(30), nn.Dropout(p=0.2)),
            'gamma': nn.Sequential(nn.Linear(self.flattened, 30), nn.LayerNorm(30), nn.Dropout(p=0.2)),
        })

        self.band_encoding_decoding_dim = {
            'delta': 1*19*37,
            'theta': 1*19*37,
            'alpha': 1*38*37,
            'beta':  1*76*37,
            'gamma': 1*114*37,
        }
        '''

        '''
        self.raw_wavelet_decoding = nn.ParameterDict({
            'delta': nn.Sequential(nn.Linear(19*11, 20), nn.Dropout(p=0.1)),
            'theta': nn.Sequential(nn.Linear(19*11, 20), nn.Dropout(p=0.1)),
            'alpha': nn.Sequential(nn.Linear(19*21, 20), nn.Dropout(p=0.1)),
            'beta':  nn.Sequential(nn.Linear(19*41, 20), nn.LayerNorm(20), nn.Dropout(p=0.1)),
            'gamma': nn.Sequential(nn.Linear(19*80, 20), nn.Dropout(p=0.1)),
        })
        '''


        '''
        self.band_encoding_decoders = nn.ParameterDict({
            'delta': nn.Sequential(nn.Linear(self.band_encoding_decoding_dim['delta'], 50), nn.LayerNorm(50), nn.Dropout(p=0.1)),
            'theta': nn.Sequential(nn.Linear(self.band_encoding_decoding_dim['theta'], 50), nn.LayerNorm(50), nn.Dropout(p=0.1)),
            'alpha': nn.Sequential(nn.Linear(self.band_encoding_decoding_dim['alpha'], 50), nn.LayerNorm(50), nn.Dropout(p=0.1)),
            'beta':  nn.Sequential(nn.Linear(self.band_encoding_decoding_dim['beta'],  50), nn.LayerNorm(50), nn.Dropout(p=0.1)),
            'gamma': nn.Sequential(nn.Linear(self.band_encoding_decoding_dim['gamma'], 50), nn.LayerNorm(50), nn.Dropout(p=0.1)),
        })
        '''
         
        #self.combined_seq = nn.Sequential(nn.Linear(1 * 19 * 10, 100), nn.LayerNorm(100), nn.Dropout(p=0.1))
        self.final_decoder = nn.Sequential(nn.GELU(), nn.Linear(3*19*10, 3*19*10), nn.LayerNorm(3*19*10), nn.GELU(), nn.Dropout(p=0.1), nn.Linear(3*19*10, 3*19*10), nn.GELU(), nn.Linear(3*19*10, 3*19*10), nn.GELU(), nn.Linear(3*19*10, 3*19*10))
        self.final_lin = nn.Sequential(nn.GELU(), nn.Linear(3*19*10, 6))
        self._init_weights()
    '''
    def forward(self, encodings):
        batch_size = encodings['delta'].shape[0]
        num_patches = encodings['delta'].shape[1]
        embedding_dim = encodings['delta'].shape[2]
        
        band_encodings_decodings = []
        for band, band_encodings in encodings.items():
            band_encodings = band_encodings.reshape(band_encodings.shape[0], self.band_encoding_decoding_dim[band])
            band_encoding_decoding = self.band_encoding_decoders[band](band_encodings)
            band_encodings_decodings.append(band_encoding_decoding)
        band_encodings_decodings = torch.cat(band_encodings_decodings, dim=1)
        x = self.combined_seq(band_encodings_decodings)
        x = self.final_lin(x)
        return x
    '''

    '''
    def forward(self, encodings, combined_manifold_output):
        batch_size = combined_manifold_output.shape[0]
        num_patches = combined_manifold_output.shape[1]
        embedding_dim = combined_manifold_output.shape[2]

        band_encodings_decodings = []
        for band, band_encodings in encodings.items():
            band_encodings = band_encodings.reshape(band_encodings.shape[0], self.band_encoding_decoding_dim[band])
            band_encoding_decoding = self.band_encoding_decoders[band](band_encodings)
            band_encodings_decodings.append(band_encoding_decoding)
        band_encodings_decodings = torch.cat(band_encodings_decodings, dim=1)

        x = self.tangent(combined_manifold_output.view(batch_size*num_patches, embedding_dim, embedding_dim))
        x = x.reshape(batch_size, num_patches, self.flattened)
        x = self.flatten(x)
        #x = self.combined_seq(x)
        #x = self.final_decoder(torch.cat([x, band_encodings_decodings], dim=1))
        #x = self.final_lin(x)
        x = torch.cat([band_encodings_decodings, x], dim=1)
        x = self.final_lin(x)
        return x
    '''

    def forward(self, combined_manifold_output):
        batch_size = combined_manifold_output.shape[0]
        num_patches = combined_manifold_output.shape[1]
        embedding_dim = combined_manifold_output.shape[2]
        x = self.tangent(combined_manifold_output.view(batch_size*num_patches, embedding_dim, embedding_dim))
        x = x.reshape(batch_size, num_patches, self.flattened)
        x = self.flatten(x)
        #x = self.combined_seq(x)
        x = self.final_decoder(x)
        x = self.final_lin(x)
        return x

    '''
    def forward(self, encodings, wavelet_manifold_output, combined_manifold_output):
        batch_size = combined_manifold_output.shape[0]
        num_patches = combined_manifold_output.shape[1]
        embedding_dim = combined_manifold_output.shape[2]

        band_encodings_decodings = []
        for band, band_encodings in encodings.items():
            band_encodings = band_encodings.reshape(band_encodings.shape[0], self.band_encoding_decoding_dim[band])
            band_encoding_decoding = self.band_encoding_decoders[band](band_encodings)
            band_encodings_decodings.append(band_encoding_decoding)
        band_encodings_decodings = torch.cat(band_encodings_decodings, dim=1)

        band_decodings = []
        for band, encodings in wavelet_manifold_output.items():
            band_tangent = self.band_tangent_spaces[band](encodings.view(batch_size*num_patches, self.band_dim[band], self.band_dim[band]))
            band_tangent = self.band_flatten[band](band_tangent)
            band_decoding = self.band_decoders[band](band_tangent)
            band_decodings.append(band_decoding)
        band_decodings = torch.cat(band_decodings, dim=1)
    x = self.tangent(combined_manifold_output.view(batch_size*num_patches, embedding_dim, embedding_dim))
    x = self.flatten(x)
    x = self.final_decoder(combined_manifold_output)
    x = self.final_lin(x)
    return x
    '''

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
