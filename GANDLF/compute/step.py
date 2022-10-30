import torch
import psutil
from .loss_and_metric import get_loss_and_metrics


def step(model, image, label, params, train=True):
    """
    Function that steps the model for a single batch

    Parameters
    ----------
    model : torch.model
        The model to process the input image with, it should support appropriate dimensions.
    image : torch.Tensor
        The input image stack according to requirements
    label : torch.Tensor
        The input label for the corresponding image label
    params : dict
        The parameters passed by the user yaml

    Returns
    -------
    loss : torch.Tensor
        The computed loss from the label and the output
    metric_output : torch.Tensor
        The computed metric from the label and the output
    output: torch.Tensor
        The final output of the model

    """
    if params["verbose"]:
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        print(
            "|===========================================================================|"
        )
        print(
            "|                              CPU Utilization                              |"
        )
        print("Load_Percent   :", psutil.cpu_percent(interval=None))
        print("MemUtil_Percent:", psutil.virtual_memory()[2])
        print(
            "|===========================================================================|"
        )

    # for the weird cases where mask is read as an RGB image, ensure only the first channel is used
    if label is not None:
        if params["problem_type"] == "segmentation":
            if label.shape[1] == 3:
                label = label[:, 0, ...].unsqueeze(1)
                # this warning should only come up once
                if params["print_rgb_label_warning"]:
                    print(
                        "WARNING: The label image is an RGB image, only the first channel will be used.",
                        flush=True,
                    )
                    params["print_rgb_label_warning"] = False

            if params["model"]["dimension"] == 2:
                label = torch.squeeze(label, -1)

    if params["model"]["dimension"] == 2:
        image = torch.squeeze(image, -1)
        if "value_keys" in params:
            if label is not None:
                if len(label.shape) > 1:
                    label = torch.squeeze(label, -1)

    if not (train) and params["model"]["type"].lower() == "openvino":
        output = torch.from_numpy(
            model(inputs={params["model"]["IO"][0][0]: image.cpu().numpy()})[
                params["model"]["IO"][1][0]
            ]
        )
        output = output.to(params["device"])
    else:
        if params["model"]["amp"]:
            with torch.cuda.amp.autocast():
                output = model(image)
        else:
            output = model(image)

    attention_map = None
    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output

    # one-hot encoding of 'label' will probably be needed for segmentation
    if label is not None:
        loss, metric_output = get_loss_and_metrics(image, label, output, params)
    else:
        loss, metric_output = None, None

    if len(output) > 1:
        output = output[0]

    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)

    return loss, metric_output, output, attention_map



def step_GAN(params, model, image, label=None):


    if params["verbose"]:
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        print(
            "|===========================================================================|"
        )
        print(
            "|                              CPU Utilization                              |"
        )
        print("Load_Percent   :", psutil.cpu_percent(interval=None))
        print("MemUtil_Percent:", psutil.virtual_memory()[2])
        print(
            "|===========================================================================|"
        )
     
        
    if params["model"]["dimension"] == 2:
        image = torch.squeeze(image, -1)
        label = torch.squeeze(label, -1)
        
    try:
        style_to_style = params["style_to_style"]
        if style_to_style==False:
            style=False
        else:
            style=True
    except KeyError:
        style=True
    if style == False:
        image = model.preprocess(image)
    else:
        image,label=model.preprocess(image,label)

    if params["model"]["amp"]:
        with torch.cuda.amp.autocast():
            if style==False:
                inp_arr = torch.zeros(image.shape[0], params["latent_dim"]).normal_(0, 1)
                output = model.return_generator()(inp_arr)
            else:
                output = model.return_generator()(image)

    else:
        print(list(model.return_generator().parameters())[0].shape)
        if style==False:
            inp_arr = torch.zeros(image.shape[0], params["latent_dim"]).normal_(0, 1)
            output = model.return_generator()(inp_arr)
        else:
            output = model.return_generator()(image)
    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output
    # one-hot encoding of 'label' will be needed for segmentation
    if style==False:
        model.set_input(image)
    else:
        model.set_input(image,label)
    model.calc_loss()
    
    loss, loss_names = model.return_loss()
    


    if len(output) > 1:
        output = output[0]

    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)

    if not ("medcam_enabled" in params and params["medcam_enabled"]):
        return loss, loss_names, output
    else:
        return loss, loss_names, output, attention_map
    
    






def step_GAN_forwardpass(model_main, image, label, params, train=False):
    from .loss_and_metric import get_loss_and_metrics_GAN as get_loss_and_metrics
    
    if params["verbose"]:
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        print(
            "|===========================================================================|"
        )
        print(
            "|                              CPU Utilization                              |"
        )
        print("Load_Percent   :", psutil.cpu_percent(interval=None))
        print("MemUtil_Percent:", psutil.virtual_memory()[2])
        print(
            "|===========================================================================|"
        )
        
        if params["model"]["dimension"] == 2:
            label = torch.squeeze(label, -1)

    if params["model"]["dimension"] == 2:
        image = torch.squeeze(image, -1)
        label = torch.squeeze(label, -1)
        
    style=True
    try:
        style_to_style = params["style_to_style"]
        if style_to_style==False:
            style=False
        else:
            style=True 
    except KeyError: 
        pass

    if style == False:
        image = torch.zeros(image.shape[0], params["latent_dim"]).normal_(0, 1)
    else:
        image,label=model_main.preprocess(image,label)

    model=model_main.return_generator()
    
    if params["model"]["amp"]:
        with torch.cuda.amp.autocast():
            output = model(image)
    else:
        output = model(image)
    if "medcam_enabled" in params and params["medcam_enabled"]:
        output, attention_map = output
  
    # one-hot encoding of 'label' will probably be needed for segmentation
    loss, metric_output = get_loss_and_metrics(image, label, output, params)

    if len(output) > 1:
        output = output[0]

    if params["model"]["dimension"] == 2:
        output = torch.unsqueeze(output, -1)
        if "medcam_enabled" in params and params["medcam_enabled"]:
            attention_map = torch.unsqueeze(attention_map, -1)
    
    if not ("medcam_enabled" in params and params["medcam_enabled"]):
        return loss, metric_output, output
    else:
        return loss, metric_output, output, attention_map
