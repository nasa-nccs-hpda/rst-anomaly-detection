# Business Case Model - Model Survey

Set of questions to walk you through decision-making scenarios for your AI/ML project's success.
This survey should serve as the first point of contact for all stakeholders. The goal is for this
survey to serve as a guidance rather than a constraint on the project's creative flexibility.

## NOTES BEFORE FILLING OUT SURVEY

/att/nobackup/nmemarsa/RST_Data/sims/ML/seg_train/*.h5
 
Examples of test data we will work with are either raw .fits files, or processed .h5 files.  Here are some examples:
 
Processed Level 1A data: /att/nobackup/nmemarsa/RST_Data/L1A/*.h5
Raw fits files: /adapt/nobackup/projects/wfirst/H4RG/HyC/20663_20666_20669_20496/*.fits

## Survey

Designed as a collection of ten questions to help guide initial meeting and/or proposal discussions,
including all four main AI/ML elements: planning, data, modeling, and operations.

### Planning

- Which scientific question would you like to address?
Detect anomalous signatures.
- Have you considered using any particular computational method? (e.g. regression, object detection,
image classification, semantic segmentation, clustering, anomaly detection, etc.)
Tried autoencoders without luck, would like to try MaskRCNN or UNet.

### Data

- Describe the data that is available, including the source, location, size, resolution, licensing, and
any training data that might be available if any.
Data is in ADAPT, and is part of instrument simulation data.

- What format is raw data in? Is there anything that must be done before the raw data may be used?
This has nothing to do with the preprocessing of specific AI/ML algorithms.
Currently H5 format.

- What aspects of the data would you like to find/address in order to solve the proposed scientific problem?
For example, land cover, signal anomalies, autonomous decisions, etc.
Anomaly detection/segmentation, currently CosmicRays

### Modeling

- Are you limited to use a particular AI/ML algorithm?
No

- Is a comparison of different algorithms to solve the scientific problem in question part of the project? Are you interested in benchmarking and optimization of algorithms?
Yes

### Operations

- Do you have enough computational resources for your problem needs? (e.g. GPUS, CPUS, TPUs, etc.)
Yes, ADAPT.

- Do you need to leverage embedded, on-premises or cloud resources? Describe any limitations and/or estimated number of computing hours if available. 
No

- Describe how the final product should look like, in particular: size, format, accessibility, static (one time only)
or dynamic (generated per request), time requirements, etc.
Output mask with CosmicRays as a first try.

### Additional Questions

- What is the timeline and funding of the project? TBD
- Any external collaborations? TBD
- Any training required before starting the project? TBD

## References

- [NASA GSFC AI Center of Excellence](https://ai.gsfc.nasa.gov/)
- [NASA Center for Climate Simulation](https://www.nccs.nasa.gov/)
- [Science Managed Cloud Environment](https://www.nccs.nasa.gov/systems/SMCE)
