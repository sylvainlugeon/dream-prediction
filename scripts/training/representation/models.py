import torch.nn as nn
import torchvision

##############################################
#           Encoder-Decoder Models           #
##############################################
    
class EncoderDecoder(nn.Module):
    
    def __init__(self, 
                 dropout, 
                 kernel_dim, 
                 encoding_dim, 
                 time_filter_dim, 
                 n_time_filters,
                 n_channels, 
                 image_dim, 
                 n_frames):

        super(EncoderDecoder, self).__init__()
        
        (self.features_extractor, 
         self.n_channels_out, 
         self.feature_map_dim) = self.get_features_extractor(kernel_dim, 
                                                        n_channels, 
                                                        image_dim)
         
                
        self.conv3D = nn.Sequential(
            nn.Conv3d(self.n_channels_out, n_time_filters, kernel_size = (time_filter_dim, 1, 1)),
        ) 
        
        assert(n_frames >= time_filter_dim)
        time_conv_dim = n_frames - (time_filter_dim - 1) # dimension after time convolution
        
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(n_time_filters * time_conv_dim * self.feature_map_dim ** 2),
            nn.Dropout(dropout),
            nn.Linear(n_time_filters * time_conv_dim * self.feature_map_dim ** 2, encoding_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, n_channels * image_dim ** 2),
        )
        
    def get_features_extractor(self, kernel_dim, n_channels, image_dim):
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
        
    def forward(self, input):
        # input: batch_size x n_frames x n_channels x 32 x 32
        
        encoding = self.encode(input)
        encoding = self.activation_fn()(encoding)
        output = self.decoder(encoding)

        return output
    
    def encode(self, input):
        # input: batch_size x n_frames x n_channels x 32 x 32
        
        batch_size, n_frames, _, _, _ = input.size()

        input = input.flatten(0, 1) # (batch_size * n_frames) x n_channels x 32 x 32
        
        # pass all frames through cnn  
        features = self.features_extractor(input) 
        features = features.reshape(batch_size, 
                                    n_frames, 
                                    self.n_channels_out, 
                                    self.feature_map_dim, 
                                    self.feature_map_dim) # batch_size x n_frames x n_filters_c3 x 4 x 4
                    
        # conv3d pooling over time (i.e frames)
        features = features.permute(0, 2, 1, 3, 4) # batch_size x n_filters_c3 x n_frames x 4 x 4
        features = self.conv3D(features) # batch_size x n_time_filters x time_conv_dim x 4 x 4

        # flatten and FC
        features = features.view(features.size(0), -1) # batch_size x (n_time_filters * time_conv_dim * 4 * 4)
        encoding = self.encoder(features)
        
        return encoding


###################################################
#           Contrastive Learning Models           #
###################################################

class ContrastiveResNet18(nn.Module):
    def __init__(self, 
                dropout, 
                encoding_dim,
                projection_dim,
                time_filter_dim, 
                n_time_filters,
                n_channels, 
                n_frames):

        super(ContrastiveResNet18, self).__init__()
        
        self.features_extractor, self.n_channels_out, self.feature_map_dim = \
            self.get_features_extractor(n_channels)
        
        assert(n_frames >= time_filter_dim)
        time_conv_dim = n_frames - (time_filter_dim - 1) # dimension after time convolution
        
        self.conv3d = nn.Conv3d(self.n_channels_out, n_time_filters, kernel_size = (time_filter_dim, 1, 1))
                
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(n_time_filters * time_conv_dim * self.feature_map_dim ** 2, encoding_dim),
            #nn.Sigmoid()
        ) 
                
        self.projection_head = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, projection_dim)
        )
        
    def get_features_extractor(self, n_channels):
        
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(n_channels, 64, 
                                 kernel_size=(7, 7), 
                                 stride=(2, 2), 
                                 padding=(3, 3), 
                                 bias=False)
        
        features_extractor = nn.Sequential(*list(resnet.children())[:-1])  
        n_out_filters = 512 # default for resnet18
        feature_map_dim = 1 # default for resnet18
        
        return features_extractor, n_out_filters, feature_map_dim
    
    def forward(self, input):
        encoding = self.encode(input)
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
        features = self.conv3d(features) # batch_size x n_time_filters x time_conv_dim x 1 x 1
        
        # flatten and FC
        encoding = self.encoder(features)
        
        return encoding
    
    def project(self, encoding):
        return self.projection_head(encoding)
        

        