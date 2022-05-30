import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import copy


##############################################
#              Baseline Models               #
##############################################

class LinearEncoderDecoder(nn.Module):
    
    def __init__(self, 
                 dropout,
                 encoding_dim,
                 n_channels,
                 image_dim,
                 n_frames):
        
        super(LinearEncoderDecoder, self).__init__()
        
        input_dim = n_channels * n_frames * image_dim ** 2
        output_dim = n_channels * image_dim ** 2
        
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, encoding_dim)
        )
            
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, output_dim)
        )
        
    def forward(self, input):
        encoding = self.encode(input)
        encoding = F.relu(encoding)
        return self.decoder(encoding)
    
    def encode(self, input):
        encoding = self.encoder(torch.flatten(input, start_dim=1))
        return encoding


##############################################
#           Encoder-Decoder Models           #
##############################################
    
class EncoderDecoder(nn.Module):
    
    def __init__(self, 
                 dropout, 
                 kernel_dim, 
                 features,
                 time_aggregation,
                 time_filter_dim, 
                 time_n_filters,
                 encoding_dim, 
                 n_decoder_layers,
                 n_channels, 
                 image_dim, 
                 n_frames):

        super(EncoderDecoder, self).__init__()
        
        assert features in {'resnet', 'vgg'}
        assert time_aggregation in {'conv', 'maxpool'}
        assert ((time_aggregation == 'conv') and time_n_filters ) or (time_aggregation != 'conv')
        
        if features == 'vgg':
            _features = vgg(kernel_dim, n_channels, image_dim)
        elif features == 'resnet':
            _features = resnet18(n_channels)
        else:
            raise NotImplementedError('Features extractor {features} does not exist')
            
        self.features_extractor, self.n_channels_out, self.feature_map_dim = _features
        
        assert(n_frames >= time_filter_dim)
         
        if time_aggregation == 'conv':     
            self.time_agg = nn.Conv3d(self.n_channels_out, time_n_filters, 
                                      kernel_size = (time_filter_dim, 1, 1))

            dim_after_time_agg = (n_frames - (time_filter_dim - 1)) * time_n_filters 
            
        if time_aggregation == 'maxpool':
            self.time_agg = nn.MaxPool3d(kernel_size = (time_filter_dim, 1, 1))
            dim_after_time_agg = int(n_frames/time_filter_dim) * self.n_channels_out
            
        dim_after_time_agg *= self.feature_map_dim ** 2
        
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(dim_after_time_agg),
            nn.Dropout(dropout),
            nn.Linear(dim_after_time_agg, encoding_dim),
        )
        
        frame_dim = n_channels * image_dim ** 2
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, frame_dim),
            *( (n_decoder_layers - 1)*[nn.ReLU(), nn.Linear(frame_dim, frame_dim)] )
        )
        
    def forward(self, input):
        # input: batch_size x n_frames x n_channels x 32 x 32
        
        encoding = self.encode(input)
        encoding = F.relu(encoding) # breaks linearity
        output = self.decoder(encoding)

        return output
    
    def encode(self, input):
        # input: batch_size x n_frames x n_channels x 32 x 32
        
        batch_size, n_frames, _, _, _ = input.size()

        input = input.flatten(0, 1) # (batch_size * n_frames) x n_channels x 32 x 32
        
        # pass all frames through features extractor  
        features = self.features_extractor(input) 
        features = features.view(batch_size, 
                                 n_frames, 
                                 self.n_channels_out, 
                                 self.feature_map_dim, 
                                 self.feature_map_dim) # batch_size x n_frames x n_filters_c3 x 4 x 4
                    
        # pooling over time (i.e frames)
        features = features.permute(0, 2, 1, 3, 4) # batch_size x n_filters_c3 x n_frames x 4 x 4
        features = self.time_agg(features) # batch_size x n_time_filters x time_conv_dim x 4 x 4

        # flatten and FC
        features = features.view(features.size(0), -1) # batch_size x (n_time_filters * time_conv_dim * 4 * 4)
        encoding = self.encoder(features)
        
        return encoding
    
    
class ConvolutionalEncoderDecoder(nn.Module):
    
    def __init__(self, 
                dropout, 
                features,
                kernel_dim,
                time_aggregation,
                time_filter_dim, 
                time_n_filters,
                encoding_dim, 
                n_channels, 
                image_dim, 
                n_frames):
        
        super(ConvolutionalEncoderDecoder, self).__init__()
        
        assert features in {'resnet', 'vgg'}
        assert time_aggregation in {'conv', 'maxpool'}
        assert ((time_aggregation == 'conv') and time_n_filters ) or (time_aggregation != 'conv')
        
        if features == 'vgg':
            _features = vgg(kernel_dim, n_channels, image_dim)
        elif features == 'resnet':
            _features = resnet18(n_channels)
        else:
            raise NotImplementedError('Features extractor {features} does not exist')
            
        self.features_extractor, self.n_channels_out, self.feature_map_dim = _features
        
        assert(n_frames >= time_filter_dim)
        
        if time_aggregation == 'conv':     
            self.time_agg = nn.Conv3d(self.n_channels_out, time_n_filters, 
                                    kernel_size = (time_filter_dim, 1, 1))

            dim_after_time_agg = (n_frames - (time_filter_dim - 1)) * time_n_filters 
            
        if time_aggregation == 'maxpool':
            self.time_agg = nn.MaxPool3d(kernel_size = (time_filter_dim, 1, 1), stride=1)
            dim_after_time_agg = (n_frames - (time_filter_dim - 1)) * self.n_channels_out
            
        dim_after_time_agg *= self.feature_map_dim ** 2
        
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_after_time_agg, encoding_dim),
        )
        
        frame_dim = n_channels * image_dim ** 2
        
        # first convt output should be 4 x 4 feature map
        first_convt_dim = 4
        first_convt_kernel = first_convt_dim - self.feature_map_dim + 1 
        assert first_convt_kernel > 0, 'Kernel size must be positive'
        
        self.pre_decoder = nn.Linear(encoding_dim, self.n_channels_out * self.feature_map_dim ** 2)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.n_channels_out, self.n_channels_out, 
                                kernel_size=first_convt_kernel, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(self.n_channels_out, 64, kernel_size=3, stride=3, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3, padding=4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_channels, kernel_size=3, stride=3, padding=8),
        )
        
        output_dim = first_convt_dim * 2 ** 3 # after the first convt, the dim is doubled three times
        self.post_decoder = nn.Linear(n_channels * output_dim ** 2, frame_dim)
        
    def forward(self, input):
        # input: batch_size x n_frames x n_channels x 32 x 32
        
        encoding = self.encode(input)
        encoding = F.relu(encoding) # breaks linearity
        
        output = self.pre_decoder(encoding)
        
        output = output.view(-1, self.n_channels_out, self.feature_map_dim, self.feature_map_dim)
        output = self.decoder(output)
        
        output = output.view(output.size(0), -1)
        output = self.post_decoder(output)

        return output
    
    def encode(self, input):
        # input: batch_size x n_frames x n_channels x 32 x 32
        
        batch_size, n_frames, _, _, _ = input.size()

        input = input.flatten(0, 1) # (batch_size * n_frames) x n_channels x 32 x 32
        
        # pass all frames through features extractor  
        features = self.features_extractor(input) 
        features = features.view(batch_size, 
                                 n_frames, 
                                 self.n_channels_out, 
                                 self.feature_map_dim, 
                                 self.feature_map_dim) # batch_size x n_frames x n_filters_c3 x 4 x 4
                    
        # pooling over time (i.e frames)
        features = features.permute(0, 2, 1, 3, 4) # batch_size x n_filters_c3 x n_frames x 4 x 4
        features = self.time_agg(features) # batch_size x n_time_filters x time_conv_dim x 4 x 4

        # flatten and FC
        features = features.view(features.size(0), -1) # batch_size x (n_time_filters * time_conv_dim * 4 * 4)
        encoding = self.encoder(features)
        
        return encoding        
        
    

###################################################
#           Contrastive Learning Models           #
###################################################

class ContrastiveLearningEncoder(nn.Module):
    def __init__(self, 
                dropout, 
                kernel_dim,
                features,
                time_aggregation,
                time_filter_dim, 
                time_n_filters,
                encoding_dim,
                projection_dim,
                n_channels, 
                n_frames,
                image_dim):

        super(ContrastiveLearningEncoder, self).__init__()
        
        assert features in {'resnet', 'vgg'}
        assert time_aggregation in {'conv', 'maxpool'}
        assert ((time_aggregation == 'conv') and time_n_filters ) or (time_aggregation != 'conv')
        
        if features == 'vgg':
            _features = vgg(kernel_dim, n_channels, image_dim)
        elif features == 'resnet':
            _features = resnet18(n_channels)
        else:
            raise NotImplementedError('Features extractor {features} does not exist')
        
        self.features_extractor, self.n_channels_out, self.feature_map_dim = _features
        
        assert(n_frames >= time_filter_dim)
        
        if time_aggregation == 'conv':     
            self.time_agg = nn.Conv3d(self.n_channels_out, time_n_filters, 
                                      kernel_size = (time_filter_dim, 1, 1))
            dim_after_time_agg = (n_frames - (time_filter_dim - 1)) * time_n_filters 

        if time_aggregation == 'maxpool':
            self.time_agg = nn.MaxPool3d(kernel_size = (time_filter_dim, 1, 1))
            dim_after_time_agg = int(n_frames/time_filter_dim) * self.n_channels_out
            
        dim_after_time_agg *= self.feature_map_dim ** 2
                
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(dim_after_time_agg, encoding_dim),
        ) 
                
        self.projection_head = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, projection_dim)
        )
    
    def forward(self, input):
        encoding = self.encode(input)
        encoding = F.relu(encoding) # non-linearity
        projection = self.projection_head(encoding)
        return projection
        
    def encode(self, input):
        # input: batch_size x n_frames x n_channels x 32 x 32
                
        batch_size, n_frames, _, _, _ = input.size()

        input = input.flatten(0, 1) # (batch_size * n_frames) x n_channels x 32 x 32
        
        # pass all frames through features extractor  
        features = self.features_extractor(input) 
        features = features.reshape(batch_size, 
                                    n_frames, 
                                    self.n_channels_out, 
                                    self.feature_map_dim, 
                                    self.feature_map_dim) # batch_size x n_frames x 512 x 1 x 1
                    
        # conv3d pooling over time (i.e frames)
        features = features.permute(0, 2, 1, 3, 4) # batch_size x 512 x n_frames x 1 x 1
        features = self.time_agg(features) # batch_size x n_time_filters x time_conv_dim x 1 x 1
        
        # flatten and FC
        encoding = self.encoder(features)
        
        return encoding
    
    def project(self, encoding):
        return self.projection_head(encoding)
    
    
###################################################
#          Model container for fine-tuning        #
###################################################
    
class FineTuner(nn.Module):
    
    def __init__(self, features_model, encoding_dim, n_classes):
        
        super(FineTuner, self).__init__()
        
        self.features_model = copy.deepcopy(features_model)
        self.clf = nn.Linear(encoding_dim, n_classes)
        
    def forward(self, x):
        encoding = self.features_model.encode(x)
        encoding = F.relu(encoding)
        output = self.clf(encoding)
        return output
        
    def encode(self, x):
        return self.features_model.encode(x)
                    
    
##############################################
#            Features extraction             #
##############################################
        

def vgg(kernel_dim, n_channels, image_dim):
    '''
    Creates and returns the main parallel CNN part of the Bashivan et al. paper from 2015. 
    The CNN is made of first: 3 Convoltional layers of 32 filters, followed by a maxpooling layer, 
    followed by 2 convolutional layers of 64 filters and a maxpooling layer, 
    followed by 1 convolutional layer of 128 filters and a maxpooling layer.
    '''
    
    # calculate the padding to keep same dimensions through convolution
    assert((kernel_dim - 1) % 2 == 0)
    padding_dim = int((kernel_dim - 1) / 2)       

    kernel_size = (kernel_dim, kernel_dim)
    padding = (padding_dim, padding_dim)
    
    max_pool_kernel_size = (2, 2)
    
    n_filters_c1 = 32
    n_filters_c2 = 64
    n_filters_c3 = 128
    
    # n_channels x 32 x 32
    
    conv1 = [
        nn.Conv2d(n_channels, n_filters_c1, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(n_filters_c1, n_filters_c1, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(n_filters_c1, n_filters_c1, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(n_filters_c1, n_filters_c1, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = max_pool_kernel_size),   
    ]
    
    # n_filters_c1 x 16 x 16
    image_dim = int((image_dim - 1) / max_pool_kernel_size[0] + 1)
    
    conv2 = [
        nn.Conv2d(n_filters_c1, n_filters_c2, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(n_filters_c2, n_filters_c2, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),         
        nn.MaxPool2d(kernel_size = max_pool_kernel_size),
    ]
    
    # n_filters_c2 x 8 x 8
    image_dim = int((image_dim - 1) / max_pool_kernel_size[0] + 1)
    
    conv3 = [
        nn.Conv2d(n_filters_c2, n_filters_c3, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = max_pool_kernel_size),
    ]
    
    # n_filters_c3 x 4 x 4
    image_dim = int((image_dim - 1) / max_pool_kernel_size[0] + 1)
    
    cnn = nn.Sequential(
        *conv1,*conv2,*conv3
    )
    
    return cnn, n_filters_c3, image_dim


def resnet18(n_in_channels):
    resnet = torchvision.models.resnet18(pretrained=False)
    resnet.conv1 = nn.Conv2d(n_in_channels, 64, 
                             kernel_size=(7, 7), 
                             stride=(2, 2), 
                             padding=(3, 3), 
                             bias=False)
    
    features_extractor = nn.Sequential(*list(resnet.children())[:-1])  
    n_out_filters = 512 # default for resnet18 on 32x32 images
    feature_map_dim = 1 # default for resnet18 on 32x32 images
    
    return features_extractor, n_out_filters, feature_map_dim