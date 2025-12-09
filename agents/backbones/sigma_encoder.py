from __future__ import annotations

import logging

log = logging.getLogger(__name__)

log.warning(
    f"Importing sigma encoder from deprecated path `agents/backbones/sigma_encoder.py.`"
)

from models.sigma_encoders import DDPMSigmaEncoder

# BackCompat
DDPM_SigmaEncoder = DDPMSigmaEncoder

# BackCompat
# The only difference between BESO_SigmaEncoder and DDPM_SigmaEncoder was the
# replacement of sigma by log(sigma)/4 in the forward pass. This has now been
# subsumed into the BesoAgent's edm_preconditioning method, so we can
# alias the two classes.
BESO_SigmaEncoder = DDPMSigmaEncoder
