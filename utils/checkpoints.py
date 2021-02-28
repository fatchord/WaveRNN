import torch
from utils.paths import Paths
from models.tacotron import Tacotron


def get_checkpoint_paths(checkpoint_type: str, paths: Paths):
    """
    Returns the correct checkpointing paths
    depending on whether model is Vocoder or TTS

    Args:
        checkpoint_type: Either 'voc' or 'tts'
        paths: Paths object
    """
    if checkpoint_type is 'tts':
        weights_path = paths.tts_latest_weights
        optim_path = paths.tts_latest_optim
        scaler_path = paths.tts_latest_scaler
        checkpoint_path = paths.tts_checkpoints
    elif checkpoint_type is 'voc':
        weights_path = paths.voc_latest_weights
        optim_path = paths.voc_latest_optim
        scaler_path = paths.voc_latest_scaler
        checkpoint_path = paths.voc_checkpoints
    else:
        raise NotImplementedError

    return weights_path, optim_path, scaler_path, checkpoint_path


def save_checkpoint(checkpoint_type: str, paths: Paths, model, optimizer, scaler, *,
        name=None, is_silent=False):
    """Saves the training session to disk.

    Args:
        paths:  Provides information about the different paths to use.
        model:  A `Tacotron` or `WaveRNN` model to save the parameters and buffers from.
        optimizer:  An optmizer to save the state of (momentum, etc).
        scaler: A scaler to save the state of (mixed precision training).
        name:  If provided, will name to a checkpoint with the given name. Note
            that regardless of whether this is provided or not, this function
            will always update the files specified in `paths` that give the
            location of the latest weights and optimizer state. Saving
            a named checkpoint happens in addition to this update.
    """
    def helper(required_path_dict, optional_path_dict, is_named):
        s = 'named' if is_named else 'latest'
        num_exist = sum(p.exists() for p in required_path_dict.values())

        if num_exist not in (0,len(required_path_dict)):
            # Checkpoint broken
            raise FileNotFoundError(
                f'We expected either both or no files in the {s} checkpoint to '
                'exist, but instead we got exactly one!')

        if num_exist == 0:
            if not is_silent: print(f'Creating {s} checkpoint...')
            for p in required_path_dict.values():
                p.parent.mkdir(parents=True, exist_ok=True)
        else:
            if not is_silent: print(f'Saving to existing {s} checkpoint...')

        if not is_silent: print(f'Saving {s} weights: {required_path_dict["w"]}')
        model.save(required_path_dict['w'])
        if not is_silent: print(f'Saving {s} optimizer state: {required_path_dict["o"]}')
        torch.save(optimizer.state_dict(), required_path_dict['o'])
        
        for p in optional_path_dict.values():
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
        
        if scaler:
            if not is_silent: print(f'Saving {s} scaler state: {optional_path_dict["s"]}')
            torch.save(scaler.state_dict(), optional_path_dict['s'])

    weights_path, optim_path, scaler_path, checkpoint_path = \
        get_checkpoint_paths(checkpoint_type, paths)

    latest_required_paths = {'w': weights_path, 'o': optim_path}
    latest_optional_paths = {'s': scaler_path}
    
    helper(latest_required_paths, latest_optional_paths, False)

    if name:
        named_required_paths = {
            'w': checkpoint_path/f'{name}_weights.pyt',
            'o': checkpoint_path/f'{name}_optim.pyt',
        }
        
        named_optional_paths = {
            's': checkpoint_path/f'{name}_scaler.pyt',
        }
        helper(named_required_paths, named_optional_paths, True)


def restore_checkpoint(checkpoint_type: str, paths: Paths, model, optimizer, scaler, *,
        name=None, create_if_missing=False):
    """Restores from a training session saved to disk.

    NOTE: The optimizer's state is placed on the same device as it's model
    parameters. Therefore, be sure you have done `model.to(device)` before
    calling this method.

    Args:
        paths:  Provides information about the different paths to use.
        model:  A `Tacotron` or `WaveRNN` model to save the parameters and buffers from.
        optimizer:  An optmizer to save the state of (momentum, etc).
        scaler: A scaler to load the state to (mixed precision training).
        name:  If provided, will restore from a checkpoint with the given name.
            Otherwise, will restore from the latest weights and optimizer state
            as specified in `paths`.
        create_if_missing:  If `True`, will create the checkpoint if it doesn't
            yet exist, as well as update the files specified in `paths` that
            give the location of the current latest weights and optimizer state.
            If `False` and the checkpoint doesn't exist, will raise a
            `FileNotFoundError`.
    """

    weights_path, optim_path, scaler_path, checkpoint_path = \
        get_checkpoint_paths(checkpoint_type, paths)

    if name:
        path_dict = {
            'w': checkpoint_path/f'{name}_weights.pyt',
            'o': checkpoint_path/f'{name}_optim.pyt',
            's': checkpoint_path/f'{name}_scaler.pyt',
        }
        s = 'named'
    else:
        required_path_dict = {
            'w': weights_path,
            'o': optim_path
        }
        optional_path_dict = {
            's': scaler_path
        }
        s = 'latest'

    num_exist = sum(p.exists() for p in required_path_dict.values())
    if num_exist == len(required_path_dict):
        # Checkpoint exists
        print(f'Restoring from {s} checkpoint...')
        print(f'Loading {s} weights: {required_path_dict["w"]}')
        model.load(required_path_dict['w'])
        print(f'Loading {s} optimizer state: {required_path_dict["o"]}')
        optimizer.load_state_dict(torch.load(required_path_dict['o']))
        
        if scaler and optional_path_dict["s"].exists():
            print(f'Loading {s} scaler state: {optional_path_dict["s"]}')
            scaler.load_state_dict(torch.load(optional_path_dict['s']))
    elif create_if_missing:
        save_checkpoint(checkpoint_type, paths, model, optimizer, scaler, name=name, is_silent=False)
    else:
        raise FileNotFoundError(f'The {s} checkpoint could not be found!')