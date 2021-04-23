---
title: "Delete all resources"
weight: 3
---

{{% notice info %}}
Start this section in a new Jupyter notebook
{{% /notice %}}

After running the demo, you should remove the resources which were created. You can also delete all the objects in the project's S3 directory by passing the keyword argument delete_s3_objects=True.


```python
from utils import delete_project_resources

delete_project_resources(
    sagemaker_boto_client=sagemaker_boto_client,
    endpoint_name=endpoint_name,
    pipeline_name=pipeline_name,
    mpg_name=mpg_name,
    prefix=prefix,
    delete_s3_objects=False,
    bucket_name=default_bucket)
```
