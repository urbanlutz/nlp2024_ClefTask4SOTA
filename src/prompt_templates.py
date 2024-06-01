
def extract_tdms_initial(tex, few_shot=True):
    if few_shot:
        return f"""If the text reports benchmark leaderboard results, extract the reported Tasks, Datasets, Metrics and corresponding Scores.
        
        Text: {tex}
        
        Return the tasks, datasets, metrics and scores as reported in the text in a JSON array. Do not include precision information in the reported score.
        Here the formating structure of the JSON. Please use exactly this formating in your answer.
        [
            {{"Task": "example Task 1", "Dataset": "example Dataset 1", "Metric": example metric 1", "Score": "score"}}, 
            {{"Task": "example Task 1", "Dataset": "example Dataset 2", "Metric": example metric 2", "Score": "score"}}
        ]
        
        Lets make an example for you: Template-Based Automatic Search of Compact Semantic Segmentation Architectures... One discovered architecture achieves 63.2% mean IoU on CamVid and 67.8% on CityScapes having only 270K parameters... evaluation.
        
        The expected answer of you is:
        [
            {{"Task": "Compact Sementic Segmentation", "Dataset": "CamVid", "Metric": Mean IoU", "Score": "63.2"}}, 
            {{"Task": "Compact Sementic Segmentation", "Dataset": "CityScapes", "Metric": Mean IoU", "Score": "67.8"}}
        ]
        
        """
    else:
        return f"""If the text reports benchmark leaderboard results, extract the reported Tasks, Datasets, Metrics and corresponding Scores.
        
        Text: {tex}
        
        Return the tasks, datasets, metrics and scores as reported in the text in a JSON array. Do not include precision information in the reported score.
        Here the formating structure of the JSON. Please use exactly this formating in your answer.
        [
            {{"Task": "example Task 1", "Dataset": "example Dataset 1", "Metric": example metric 1", "Score": "score"}}, 
            {{"Task": "example Task 1", "Dataset": "example Dataset 2", "Metric": example metric 2", "Score": "score"}}
        ]
        
        """
    
def extract_tdms_optimized01(tex, few_shot=True):
    if few_shot:
        return f"""I am a researcher and I want you to extract benchmark leaderboard results from the provided sequence of a scholarly article about Artificial Intelligence. Please return a JSON object with the defined structure below. I expect you to find TDMS(task, dataset, metric, score) tuples. Do not include precision information in the reported score. Please have a look at the example and the expected result below as well. Thank you.

        JSON structure:
        [
            {{"Task": "example Task 1", "Dataset": "example Dataset 1", "Metric": example metric 1", "Score": "score"}}, 
            {{"Task": "example Task 1", "Dataset": "example Dataset 2", "Metric": example metric 2", "Score": "score"}}
        ]
        
        Example:
        Template-Based Automatic Search of Compact Semantic Segmentation Architectures... One discovered architecture achieves 63.2% mean IoU on CamVid and 67.8% on CityScapes having only 270K parameters... evaluation.
        
        Result:
        [
            {{"Task": "Compact Sementic Segmentation", "Dataset": "CamVid", "Metric": Mean IoU", "Score": "63.2"}}, 
            {{"Task": "Compact Sementic Segmentation", "Dataset": "CityScapes", "Metric": Mean IoU", "Score": "67.8"}}
        ]
        
        Sequence: 
        {tex}
        
        """
    else:
        return f"""I am a researcher and I want you to extract benchmark leaderboard results from the provided sequence of a scholarly article about Artificial Intelligence. Please return a JSON object with the defined structure below. I expect you to find TDMS(task, dataset, metric, score) tuples. Do not include precision information in the reported score. Thank you.

        JSON structure:
        [
            {{"Task": "example Task 1", "Dataset": "example Dataset 1", "Metric": example metric 1", "Score": "score"}}, 
            {{"Task": "example Task 1", "Dataset": "example Dataset 2", "Metric": example metric 2", "Score": "score"}}
        ]
        
        
        Sequence: 
        {tex}
        
        """
    
def extract_tdms_optimized02(tex, few_shot=True):
    if few_shot:
        return f"""I am a researcher and I want you to extract benchmark leaderboard results from the provided sequence of a scholarly article about Artificial Intelligence. Please return a JSON object with the defined structure below. I expect you to find TDMS(task, dataset, metric, score) tuples. Most common tasks are image classification, atari games, node classification, object detection, video retrieval, link prediction, semantic segmentation, semi-supervised video object segmentation, 3d human pose estimation and question answering. Most common datasets are imagenet, coco test-dev, human3.6m, cifar-10, coco minival, youtube-vos 2018, cifar-100, msr-vtt-1ka, fb15k-237 and msu super-resolution for video compression. Most common metrics are accuracy, score, f1, psnr, map, miou, ssim, top 1 accuracy, 1:1 accuracy and number of params. Do not include precision information in the reported score. Please have a look at the example and the expected result below as well. Thank you.

        JSON structure:
        [
            {{"Task": "example Task 1", "Dataset": "example Dataset 1", "Metric": example metric 1", "Score": "score"}}, 
            {{"Task": "example Task 1", "Dataset": "example Dataset 2", "Metric": example metric 2", "Score": "score"}}
        ]
        
        Example:
        Template-Based Automatic Search of Compact Semantic Segmentation Architectures... One discovered architecture achieves 63.2% mean IoU on CamVid and 67.8% on CityScapes having only 270K parameters... evaluation.
        
        Result:
        [
            {{"Task": "Compact Sementic Segmentation", "Dataset": "CamVid", "Metric": Mean IoU", "Score": "63.2"}}, 
            {{"Task": "Compact Sementic Segmentation", "Dataset": "CityScapes", "Metric": Mean IoU", "Score": "67.8"}}
        ]
        
        Sequence: 
        {tex}
        
        """
    else:
        return f"""I am a researcher and I want you to extract benchmark leaderboard results from the provided sequence of a scholarly article about Artificial Intelligence. Please return a JSON object with the defined structure below. I expect you to find TDMS(task, dataset, metric, score) tuples. Most common tasks are image classification, atari games, node classification, object detection, video retrieval, link prediction, semantic segmentation, semi-supervised video object segmentation, 3d human pose estimation and question answering. Most common datasets are imagenet, coco test-dev, human3.6m, cifar-10, coco minival, youtube-vos 2018, cifar-100, msr-vtt-1ka, fb15k-237 and msu super-resolution for video compression. Most common metrics are accuracy, score, f1, psnr, map, miou, ssim, top 1 accuracy, 1:1 accuracy and number of params. Do not include precision information in the reported score. Thank you.

        JSON structure:
        [
            {{"Task": "example Task 1", "Dataset": "example Dataset 1", "Metric": example metric 1", "Score": "score"}}, 
            {{"Task": "example Task 1", "Dataset": "example Dataset 2", "Metric": example metric 2", "Score": "score"}}
        ]
        
        Sequence: 
        {tex}
        
        """


zero_shot_template_initial = lambda tex: extract_tdms_initial(tex, False)
few_shot_template_initial = lambda tex: extract_tdms_initial(tex, True)
zero_shot_template_optimized01 = lambda tex: extract_tdms_optimized01(tex, False)
few_shot_template_optimized01 = lambda tex: extract_tdms_optimized01(tex, True)
zero_shot_template_optimized02 = lambda tex: extract_tdms_optimized02(tex, False)
few_shot_template_optimized02 = lambda tex: extract_tdms_optimized02(tex, True)
