import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]
            
        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=10, n_hidden_neurons=256, n_layers=8):
        super().__init__()
        
        # REDUCED harmonic functions from 60 to 10 to prevent high-freq noise
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)
        
        # The dimension of the harmonic embedding.
        embedding_dim = n_harmonic_functions * 2 * 3
        self.input_dim = embedding_dim
        
        # Main MLP layers
        self.n_layers = n_layers
        self.skip_layer = n_layers // 2  # Add input back in at the middle layer
        
        layers = []
        for i in range(n_layers):
            if i == 0:
                dim_in = embedding_dim
                dim_out = n_hidden_neurons
            elif i == self.skip_layer:
                dim_in = n_hidden_neurons + embedding_dim # Skip connection input
                dim_out = n_hidden_neurons
            else:
                dim_in = n_hidden_neurons
                dim_out = n_hidden_neurons
            
            layers.append(torch.nn.Linear(dim_in, dim_out))
        
        self.mlp_layers = torch.nn.ModuleList(layers)
        
        # switched Softplus to ReLU (standard for NeRF, sharper results)
        self.activation = torch.nn.ReLU() 

        # Color branch
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons // 2, 3),
            torch.nn.Sigmoid(),
        )  
        
        # Density branch
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0), # Softplus is fine for density to keep it positive
        )
        
        # Initialize density bias to be low (empty space)
        torch.nn.init.constant_(self.density_layer[0].bias, -1.5)        
                
    def _get_densities(self, features):
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()
    
    def _get_colors(self, features, rays_directions):
        # Re-compute embedding for view dependence (simplified for this fix)
        # Note: In a full implementation, you usually want a separate direction embedding
        # but using the spatial embedding here is a passable simplification for now.
        
        # Ideally, we pass the direction embedding here, but to keep your code structure:
        spatial_size = features.shape[:-1]
        
        # Recalculate embedding just for concatenation sizing (or reuse if passed)
        # For this specific fix, we will just reuse the features + input embedding strategy
        # used in the forward pass below.
        return self.color_layer
    
    def forward(self, ray_bundle: RayBundle, **kwargs):
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        
        # 1. Embed inputs
        embeds = self.harmonic_embedding(rays_points_world)
        
        # 2. Run through deep MLP with Skip Connection
        h = embeds
        for i, layer in enumerate(self.mlp_layers):
            if i == self.skip_layer:
                # Concatenate input embedding back into the network
                h = torch.cat([h, embeds], dim=-1)
            h = layer(h)
            h = self.activation(h)
            
        features = h
        
        # 3. Outputs
        rays_densities = self._get_densities(features)
        
        # For color, we concatenate the features with the spatial embedding
        # (Standard NeRF concatenates Direction embedding, but your code used Spatial)
        color_input = torch.cat([features, embeds], dim=-1)
        rays_colors = self.color_layer(color_input)
        
        return rays_densities, rays_colors

    # Keep your batched_forward exactly as it was...
    def batched_forward(self, ray_bundle: RayBundle, n_batches: int = 16, **kwargs):
        # ... (Paste your existing batched_forward code here) ...
        n_pts_per_ray = ray_bundle.lengths.shape[-1]  
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors