---
title: "Setup Amazon SageMaker Studio"
weight: 1 
---

## Step 1.  Log in to the Amazon SageMaker console

1. Open the [AWS Management Console](https://console.aws.amazon.com/console/home) in a new window, so you can keep this tutorial open.
2. In the AWS Console search bar, type _SageMaker_ and select Amazon SageMaker to open the service console.

![](/images/image001.png)

## Step 2.  Set up Amazon SageMaker Studio

In this step, you&#39;ll setup Amazon SageMaker Studio. The SageMaker Studio Notebooks within Studio

are one-click Jupyter notebooks and contain everything you need to build and test your training scripts. Studio also includes experiment tracking and visualization so that it&#39;s easy to manage your entire machine learning workflow in one place.

1. Navigate to Amazon SageMaker Console \&gt; Amazon SageMaker Studio
![](/images/image002.png)

2. If this is your first time using Amazon SageMaker Studio, you must complete the [Studio onboarding process](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html). When onboarding, you can choose to use either [AWS Single Sign-On (AWS SSO)](https://aws.amazon.com/single-sign-on/) or [AWS Identity and Access Management (IAM)](https://aws.amazon.com/iam/) for authentication methods. When you use IAM authentication, you can choose either the Quick start or the Standard setup procedure. If you are unsure of which option to choose, see [Onboard to Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) and ask your IT administrator for assistance. For simplicity, this tutorial uses the Quick start procedure.
2. In the  **Get started**  box, choose  **Quick start**. Leave the default name as it is.
![](/images/image003.png)

1. Under **Execution role** , choose  **Create an IAM role**. In the dialog box that appears, choose  **Any S3 bucket**  and choose Create role. Amazon SageMaker creates a role with the required permissions and assigns it to your instance. This allows your Amazon SageMaker instance to access all the Amazon S3 buckets in your account. If you already have a bucket that you&#39;d like to use instead, select Specific S3 buckets and specify the bucket name.
![](/images/image004.png)

1. Select Create role.
 Amazon SageMaker creates the _AmazonSageMaker-ExecutionRole-\*\*\*_ role.
![](/images/image005.png)

1. Click **Submit.** You should see a message saying that the studio is being configured.
![](/images/image006.png)

1. When Amazon SageMaker Studio is ready, click **Open Studio**![](RackMultipart20200929-4-54c2ot_html_eafe0c3b822e3103.png)
2. Amazon SageMaker Studio should now be open on a separate browser tab.
![](/images/image007.png)

