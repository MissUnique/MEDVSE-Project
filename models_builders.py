import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(16)

        self.conv4 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(24)

        self.dropout = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(864, 32)
        self.dense2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, channels, sequence_length]

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)

        x = self.dropout(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)

        return x


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=3, stride=1)

        self.conv3 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=3, stride=1)

        self.conv4 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=3, stride=1)

        self.conv5 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=3, stride=1)

        self.conv6 = nn.Conv1d(in_channels=32, out_channels=36, kernel_size=3, stride=1)

        self.conv7 = nn.Conv1d(in_channels=36, out_channels=1, kernel_size=3, stride=1)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))

        x = F.relu(self.conv6(x))

        x = F.relu(self.conv7(x))

        x = self.global_avg_pool(x)

        # Remove the last dimension
        x = x.squeeze(-1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, inp_filt, out_filt):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(inp_filt, out_filt, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv1d(out_filt, inp_filt, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        identity = x
        out = F.elu(self.conv1(x))
        out = F.elu(self.conv2(out))
        return identity + out


class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv1d(3, 8, kernel_size=3, stride=1, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.residual_block1 = ResidualBlock(8, 16)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding='same')

        self.residual_block2 = ResidualBlock(16, 24)
        self.conv3 = nn.Conv1d(16, 24, kernel_size=3, stride=1, padding='same')

        self.residual_block3 = ResidualBlock(24, 32)
        self.conv4 = nn.Conv1d(24, 32, kernel_size=3, stride=1, padding='same')

        self.residual_block4 = ResidualBlock(32, 48)
        self.conv5 = nn.Conv1d(32, 48, kernel_size=3, stride=1, padding='same')

        self.residual_block5 = ResidualBlock(48, 64)
        self.conv6 = nn.Conv1d(48, 64, kernel_size=3, stride=1, padding='same')

        self.conv7 = nn.Conv1d(64, 1, kernel_size=3, stride=1, padding='same')
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = F.elu(self.conv1(x))
        x = self.pool(x)

        # Residual block 16
        x = self.residual_block1(x)
        x = F.elu(self.conv2(x))

        # Residual block 24
        x = self.residual_block2(x)
        x = F.elu(self.conv3(x))

        # Residual block 32
        x = self.residual_block3(x)
        x = F.elu(self.conv4(x))

        # Residual block 48
        x = self.residual_block4(x)
        x = F.elu(self.conv5(x))

        # Residual block 64
        x = self.residual_block5(x)
        x = F.elu(self.conv6(x))

        x = self.conv7(x)
        x = self.global_avg_pool(x)

        x = x.squeeze(-1)
        return x


class convnext_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(convnext_block, self).__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=7,
            padding="same",
            groups=in_channels,
            stride=1
        )
        self.norm_layer = nn.LayerNorm(in_channels)
        self.pointwise_conv_1 = nn.Conv1d(in_channels, int(1.5 * out_channels), kernel_size=1)
        self.pointwise_conv_2 = nn.Conv1d(int(1.5 * out_channels), in_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        # Depthwise conv
        x = self.depthwise_conv(x)  # [batch, channel, sequence_length]

        x = x.permute(0, 2, 1)  # [batch, sequence_length, channel]
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)  # [batch, channel, sequence_length]

        x = F.gelu(self.pointwise_conv_1(x))
        x = self.pointwise_conv_2(x)

        return x + residual


class ModifiedConvnextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.convnext_block_16 = convnext_block(in_channels=8, out_channels=16)
        self.convnext_block_24 = convnext_block(in_channels=16, out_channels=24)
        self.convnext_block_32 = convnext_block(in_channels=24, out_channels=32)
        self.convnext_block_48 = convnext_block(in_channels=32, out_channels=48)
        self.convnext_block_64 = convnext_block(in_channels=48, out_channels=64)

        self.conv_2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv1d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.conv_6 = nn.Conv1d(in_channels=48, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Expect input shape: [batch, time_steps, features]
        x = x.permute(0, 2, 1)  # Convert to [batch, features, time_steps]

        x = self.conv_1(x)

        x = self.convnext_block_16(x)
        x = F.gelu(self.conv_2(x))

        x = self.convnext_block_24(x)
        x = F.gelu(self.conv_3(x))

        x = self.convnext_block_32(x)
        x = F.gelu(self.conv_4(x))

        x = self.convnext_block_48(x)
        x = F.gelu(self.conv_5(x))

        x = self.convnext_block_64(x)
        x = F.gelu(self.conv_6(x))

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)

        return x