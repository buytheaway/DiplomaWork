# 1. Introduction

The rapid growth of digital services, access control infrastructure, and security analytics systems has sharply increased the demand for reliable and fast biometric identification. In many practical scenarios a system is not asked to classify a face into a fixed, closed set of classes. Instead, it must search a large repository and return the most likely candidates for further operator review. This difference is crucial. A classification problem can be solved with a conventional classifier trained for a known set of identities, while an identification search problem requires a flexible similarity-based architecture capable of handling new persons, new embeddings, and large-scale updates over time [1], [2].

Face biometrics are especially attractive because the acquisition process is contactless and can be integrated into ordinary cameras, desktop applications, and operator workstations. However, face search also introduces several technical challenges: the variability of lighting and pose, search latency at scale, privacy and retention constraints, and the need to support operational workflows such as enrollment, search, index rebuild, and audit logging [3], [4]. A research project in this area should therefore not be limited to a neural network alone. It should describe a complete applied system in which machine learning, vector search, storage, user interface design, and deployment strategy are treated as parts of the same solution.

This diploma project addresses that exact problem. The project, titled **Development of Algorithms and Methods for Fast Search in Biometric Databases**, implements a modular biometric face search system with a FastAPI backend, a FAISS-based vector search layer, relational metadata storage, and a PySide6 desktop application for operational use. The project combines two logical model branches. The first branch is a pretrained baseline used for stable runtime operation and controlled comparison. The second branch is a custom PyTorch research extension. The stable MVP relies on pretrained ONNX or InsightFace embedding extraction; custom model quality claims require valid training logs and labeled verification results before they can be used as biometric accuracy evidence.

The practical goal of the project is to design and implement a system that can:

- enroll a person from an image without storing the original face image by default;
- perform fast similarity search over a biometric repository;
- maintain separate logical search pipelines for comparison;
- support multiple operational modes, including image-based search, compare mode, and live webcam-assisted search;
- provide reproducible architecture, experiments, and deployment artifacts suitable for pre-defense and further diploma refinement.

The scientific and engineering relevance of the project is determined by several factors. First, approximate nearest-neighbor search over learned embeddings is a central method in modern large-scale biometric systems. Second, practical face search solutions must satisfy privacy constraints and data minimization principles, especially when used in controlled environments. Third, researchers and developers need transparent system architectures that make it possible to tune detectors, embedding models, and FAISS indices independently [5], [6].

The object of research is a biometric identification system based on facial images. The subject of research is the set of methods and architectural solutions that provide fast similarity search, modular runtime composition, and practical operator interaction in a desktop client. The project focuses on a prototype that is strict where required, modular by design, and reproducible enough to support both operational demonstration and academic explanation.

The main objective of the diploma project is to develop algorithms and methods for fast search in biometric databases and to implement them in a working prototype. To achieve this objective, the project solves the following tasks:

- analyze the state of face-based biometric identification systems and the role of embedding-based similarity search;
- compare relevant technologies for API implementation, desktop interaction, vector indexing, and storage;
- design a modular architecture with explicit separation between API, preprocessing, vector search, and metadata storage;
- implement a backend service and a desktop client;
- support both pretrained and custom model branches in a comparative runtime design;
- prepare architecture diagrams, process models, and implemented interface screenshots for pre-defense presentation;
- formulate practical conclusions and limitations of the current prototype.

The methodology of the work combines software architecture design, applied machine learning system engineering, and comparative technology analysis. The project uses a ports-and-adapters style organization inside the backend service, FAISS for vector search, SQLAlchemy-based repositories for persistence, PySide6 for the operator desktop, and PyTorch/ONNX branches for model-related experimentation. The architectural explanation is strengthened by C4-style diagrams, BPMN process views, sequence diagrams, and UI screenshots extracted from the running system and supporting design materials.

The novelty of the project lies not in the claim of inventing a completely new biometric algorithm, but in the integrated design of a practical fast-search prototype that combines embedding-based biometric search, explicit separation of storage and vector search responsibilities, dual-pipeline comparison, a desktop operator interface, and an extendable research branch for custom model work. This makes the project suitable both as a technical prototype and as a coherent diploma case.

From a practical perspective, the system can be adapted for access control support, archive search, operator-assisted biometric review, and controlled experiments with different vector index settings. From an academic perspective, it provides a strong basis for discussing trade-offs among latency, top-k retrieval overlap, privacy, maintainability, and deployment complexity.

[[PAGE_BREAK]]
# 2. Literature Review

## 2.1. Biometric Identification as an Applied Research Field

Biometric systems are designed to identify or verify a person based on physiological or behavioral features such as fingerprints, iris, voice, gait, or facial appearance. Among these modalities, face biometrics occupy a special position because they are relatively easy to acquire, can be integrated into ordinary cameras, and support contactless interaction. This convenience explains the wide adoption of face-based systems in access control, customer identity verification, public safety analytics, border control, and digital onboarding [1], [3].

At the same time, facial biometrics present a difficult engineering problem. The same person may look different due to illumination, occlusion, head pose, image resolution, aging, facial hair, cosmetics, camera noise, and expression changes. Therefore, robust face search depends not only on the recognition model itself but also on the end-to-end processing pipeline: input validation, detection, alignment, quality control, embedding extraction, similarity computation, thresholding, ranking, and operator review [2], [4].

The literature usually distinguishes between verification and identification. Verification answers the question “Is this person who they claim to be?” Identification answers the question “Who is this person among the entries stored in a database?” The second case is more challenging because the system must compare a probe image with many stored vectors and return the nearest candidates with sufficiently low latency [2], [7]. The present diploma belongs to the identification-search category.

## 2.2. Evolution of Face Recognition Methods

Early face recognition systems relied on handcrafted feature representations such as Eigenfaces, Fisherfaces, Local Binary Patterns (LBP), Histogram of Oriented Gradients (HOG), and similar descriptors [8], [9]. These methods were computationally attractive and important historically, but they tended to degrade when the visual conditions changed significantly. In addition, such handcrafted descriptors were not ideal for very large-scale nearest-neighbor search because their discriminative power was lower than that of modern deep representations.

The emergence of deep learning transformed the field. Convolutional neural networks made it possible to learn compact face embeddings in which images of the same person are close to one another and images of different people are separated in vector space. Rather than building a classifier whose last layer corresponds to a fixed set of people, embedding-based approaches learn a general face representation that can be used for similarity search on unseen identities [10], [11].

Among the most influential deep face recognition works are FaceNet and ArcFace. FaceNet introduced the idea of learning an embedding using triplet loss, directly optimizing a metric space in which Euclidean distance corresponds to identity similarity [10]. ArcFace later improved discriminative power by introducing an additive angular margin, leading to stronger separation between identities in normalized embedding space [11]. These works, together with later open-source ecosystems such as InsightFace, shaped the current mainstream approach to practical face search systems.

## 2.3. Face Detection and Alignment

Recognition quality is strongly tied to the quality of detection and alignment. A recognition model may be powerful, but if the input crop is badly localized, rotated, or includes too much irrelevant background, the resulting embedding may be unstable. This is why modern pipelines typically begin with a dedicated face detector, often with landmark support. Landmark-based alignment helps normalize the relative position of eyes, nose, and mouth before embedding extraction [4], [12].

RetinaFace and MTCNN are widely referenced detectors in the literature and practical toolchains. The idea is not simply to find a rectangle around a face but to create a stable geometric representation that reduces nuisance variation. This is especially important in systems that must compare many vectors quickly, because vector search quality can only be as good as the embeddings sent into the index.

An important operational policy follows from this observation: the system should not “guess” when the input is ambiguous. In this project, enrollment is intentionally strict and requires exactly one face in the image. Search is more permissive and supports multiple faces, because search often deals with scene images or frames containing more than one person. This distinction reflects a practical interpretation of the literature: data quality rules should depend on the business operation being performed.

## 2.4. Embeddings and Metric Learning

The dominant paradigm in modern face search is to transform each face into a dense numeric vector, often of dimensionality 128, 256, or 512. Once faces are represented as vectors, identification becomes a nearest-neighbor search problem. Similarity can be measured with cosine similarity, inner product, or Euclidean distance, depending on whether the embeddings are normalized and how the index is configured [5], [10], [11].

Metric learning losses attempt to enforce a desirable geometry in embedding space. Triplet loss encourages an anchor to be closer to a positive sample than to a negative sample by at least a margin. Angular-margin losses such as ArcFace reshape the classification objective so that the embedding space becomes more discriminative and more suitable for recognition tasks [10], [11]. The practical advantage of this approach is that new people can be added to the gallery without retraining a closed-set classifier for every deployment scenario.

For this diploma project, the embedding-based formulation is central. It allows the system to maintain a repository of biometric vectors, update the repository incrementally, and perform search through approximate nearest-neighbor indices. It also supports a meaningful comparison between a stable pretrained baseline and an experimental custom branch.

## 2.5. Approximate Nearest-Neighbor Search and FAISS

If a system stores only a few hundred vectors, brute-force comparison is still manageable. However, once the gallery grows to tens or hundreds of thousands of embeddings, naive linear scan becomes progressively more expensive. This motivates the use of approximate nearest-neighbor search methods. The research literature and engineering practice offer several families of such methods, including graph-based search, inverted files, product quantization, and hybrid approaches [5], [13], [14].

FAISS is one of the most influential toolkits in this domain. Developed by Meta, it supports exact flat indices, graph-based indices such as HNSW, and compressed indices such as IVF-PQ [5]. Its importance for this project is twofold. First, it offers high-performance search over dense vectors. Second, it gives direct access to algorithm parameters, which is valuable for controlled academic experimentation. Instead of outsourcing search behavior to a black-box platform, a student project can explicitly discuss the trade-offs among latency, top-k retrieval overlap, memory use, and rebuild complexity.

## 2.6. Client Interaction and Human-in-the-Loop Review

Many real biometric systems are not fully autonomous. They are operator-assisted. A search system may return a ranked list of candidates, but a human still reviews the result, checks context, and makes the final decision. This is especially common in compliance-sensitive domains or in situations where the cost of false acceptance is high. Therefore, the quality of the operator interface matters almost as much as backend latency [3], [15].

Desktop applications remain relevant in controlled enterprise environments because they support stable workstation workflows, local hardware access, and clearer operational boundaries than generic consumer-facing web interfaces. A desktop client can integrate image loading, result review, compare mode, live webcam preview, and log inspection in a single environment. A web-only control panel would be possible, but for this project the desktop format is more appropriate for an operator workstation.

## 2.7. Privacy, Security, and Data Minimization

Biometric data is sensitive. The literature on biometric privacy and secure template handling emphasizes that raw biometric artifacts can be harder to revoke than passwords and may create long-term governance issues if stored unnecessarily [16]. As a result, data minimization is not just a legal or ethical preference; it is an important architectural constraint.

This diploma adopts a clear data minimization rule: original face images are not stored by default. Images are used only to compute embeddings and then discarded, while the system stores embeddings, person metadata, and index snapshots. This reduces the risk surface, lowers storage requirements, and keeps the prototype aligned with privacy-aware design principles.

## 2.8. Literature Review Conclusion

The literature review shows that a modern face search system is best understood as an integration problem. High-quality recognition depends on detection and alignment; large-scale performance depends on approximate nearest-neighbor search; practical deployment depends on API design, storage choices, and user interface quality; and responsible operation depends on privacy-aware data handling. These findings directly shape the project architecture and explain the selected technology stack.

[[PAGE_BREAK]]
# 3. Analysis of Existing Systems

## 3.1. Closed Enterprise Biometric Access Systems

Closed enterprise biometric systems are common in access control, internal facility management, and restricted workstation environments. Their main strengths are operational predictability, stable hardware conditions, and integration with local policies. Such systems usually combine cameras, user directories, access rules, and event logging. They often work well when the number of users is fixed or the deployment is tailored to a single enterprise scenario.

However, many of these systems are not designed as open research platforms. Their model configuration, vector indexing strategy, and evaluation procedures may be hidden from the operator. This makes them less suitable for academic experimentation. A diploma project requires not only a functioning result but also a transparent explanation of why each subsystem exists and what trade-offs it introduces.

## 3.2. Cloud API Recognition Services

Cloud-based image analysis and recognition services offer a different model. They are easy to start with, expose HTTP APIs, and often provide convenient scaling. A developer can send an image and receive labels, matches, or metadata with minimal infrastructure work. These services are attractive for prototypes, but they also introduce several limitations from the perspective of this diploma.

First, data governance becomes more complex when biometric data is sent to external infrastructure. Second, experimentation is constrained by the service provider’s API surface. Third, latency and deployment assumptions depend on network conditions and cloud policy. For a diploma project focused on architecture, reproducibility, and controlled experiments, such external dependence is a significant drawback. It is more appropriate to use local runtimes and explicit indices whose behavior can be explained and tuned by the team.

## 3.3. Open-Source Face Recognition Stacks

Open-source stacks such as InsightFace, facenet-based repositories, and various PyTorch face-recognition implementations provide a strong technical basis for experiments. They usually expose model definitions, preprocessing logic, and training scripts, which makes them useful for research. They also encourage reproducibility and allow the student to explain architectural decisions at a finer level of detail than a commercial black box would permit.

At the same time, open-source toolchains often focus on only one part of the problem. Some emphasize model training, some emphasize detection, and some emphasize inference. They may not provide a complete operator-facing system with metadata management, desktop interaction, index rebuild workflows, and logging. This gap is precisely where the current diploma project situates itself: not as a single-model repository, but as an integrated biometric search prototype.

## 3.4. Vector Databases and Search Platforms

Vector databases such as Milvus, Qdrant, Weaviate, and similar systems have popularized dense-vector retrieval in production environments. They provide persistence, indexing, and API access in one platform. For some projects this is an excellent choice. However, the current diploma favors direct FAISS usage because it offers clearer access to index internals and parameter-level experimentation.

This does not mean vector databases are inferior. Rather, it reflects the goals of the work. A thesis prototype benefits from being able to explain exactly how index construction, memory usage, and approximate nearest-neighbor behavior are controlled. If a vector database abstracts too much of that behavior away, the system becomes easier to deploy but harder to analyze as an academic artifact.

## 3.5. Comparative Gap Analysis

The analysis of existing systems leads to several conclusions:

- closed enterprise systems are operationally practical but often insufficiently transparent for research;
- cloud recognition services are convenient but less suitable for privacy-sensitive, on-premise, and reproducible experimentation;
- open-source model repositories are useful building blocks but rarely provide a full operator system;
- vector platforms simplify infrastructure but may hide details that are valuable in a thesis.

Therefore, the proposed project occupies a useful middle ground. It combines open technologies with a complete applied workflow, remains deployable locally, and still exposes enough architectural detail to support formal explanation and critical analysis.

[[PAGE_BREAK]]
# 4. Data Collection

## 4.1. Data Requirements for Biometric Search

Data collection in biometric search is not merely a matter of downloading images. The system must define what constitutes a valid identity sample, what the structure of the dataset should be, which images are acceptable for enrollment, and how privacy constraints are enforced. In this project, the data pipeline is identity-oriented: each person corresponds to a set of face images that may later be transformed into embeddings and inserted into the search repository.

The project follows a directory-oriented convention for externally prepared datasets:

- `dataset/<identity>/*.jpg`
- or a train/validation/test split such as `datasets/<name>/train/<identity>/*.jpg`

This structure is easy to validate, works well with PyTorch dataset loaders, and reflects the logic of identity-based recognition tasks. It also avoids hiding critical assumptions in opaque preprocessing tools.

## 4.2. External Datasets and Research Inputs

The project is designed to accept external datasets rather than bundle raw training corpora into the repository. This is important for both licensing and practical reasons. Public face datasets may have specific usage restrictions, and large image collections should not be committed directly into the code repository. The codebase instead provides scripts and loaders that transform external inputs into embeddings and index-ready artifacts.

For the final diploma version, the dataset roles are fixed. The main real-face training dataset for the custom IR-50 experiments is CelebA prepared as identity folders under `datasets/celeba_faces/train`, with validation data under `datasets/celeba_faces/val`. LFW is not the training dataset; it is reserved for final labeled-pair verification and threshold analysis through `handoff_lfw_eval/lfw` and `handoff_lfw_eval/lfw/pairs.txt`. Synthetic 512-dimensional vectors are used only for retrieval scalability experiments and do not replace biometric quality evaluation. Optional synthetic or generated sources such as DigiFace1M can be used for warmstart or scale simulation, but they are not presented as the final real-face biometric validation dataset.

The important point is not to claim that all datasets are stored in the repository, but to show that the system defines a reproducible protocol: external data enters through a controlled folder structure, is checked, converted into embeddings, and searched through FAISS. This is consistent with the research-ready goals of the project.

## 4.3. Enrollment Data Policy

Enrollment quality rules are stricter than search rules. In an enrollment operation, the system expects exactly one face. If no face is found, enrollment must fail. If multiple faces are found, enrollment must also fail because assigning one label to multiple distinct faces would corrupt the repository. This is a deliberate design decision grounded in both engineering logic and data integrity.

Search differs from enrollment because it is often used on group photos or scene frames. Therefore, search in this project supports multiple faces and processes them separately. This distinction is not a minor UI choice; it is a data governance rule. It ensures that the gallery remains clean while still allowing practical operational search behavior.

## 4.4. Quality Filtering and Data Preparation

Data preparation for biometric search involves more than naming folders. Before an image can become a stable embedding, the system needs to verify that it can decode the image, detect a face, assess that the detection is usable, and align the crop appropriately. Low-quality or ambiguous images must be rejected or at least flagged for review. Even when a project does not build a full enterprise-grade data curation pipeline, it must define the quality gates clearly.

The present project implements these gates through the backend face-processing path. Image bytes are decoded, faces are detected, and only then are embeddings produced. The storage layer never becomes the first place where quality problems are discovered. This improves robustness and keeps the database cleaner.

## 4.5. Privacy Rule: Images Are Not Stored by Default

One of the most important data decisions in the project is the explicit refusal to store raw input face images by default. This choice influences architecture, deployment, privacy posture, and even how the project is described academically. Instead of keeping a permanent gallery of original images, the system stores the following long-term artifacts:

- person metadata;
- face embeddings as numeric vectors;
- FAISS snapshot metadata;
- operational logs and index status information.

This rule reduces the storage footprint and aligns the project with data minimization principles. It also creates a clear narrative for the diploma: the project processes biometric data responsibly and avoids unnecessary retention of sensitive raw images.

## 4.6. The Pretrained and Custom Model Data Narrative

The project combines two model lines. The pretrained line is used as the stable baseline. It operates through ONNX or InsightFace runtime assets and serves as the main reliable reference for comparison and operational demonstration. The custom line is treated as an experimental PyTorch research branch. The repository contains training code, model architecture, evaluation utilities, and local checkpoint support, but the current documentation does not claim final biometric accuracy for this branch.

This dual-model narrative is valuable because it separates operational stability from research extensibility. The pretrained path supports the stable MVP demonstration, while the custom branch shows how the system can be extended with a project-specific model after valid training logs and labeled verification results are available.

## 4.7. Database-Oriented Storage of Embeddings

The project stores embeddings and metadata in a relational database while leaving nearest-neighbor search to FAISS. This separation is a deliberate data management decision. The database is used for persistence, identifiers, labels, and history-related entities, while the FAISS index serves as the operational search engine. This prevents the system from misusing a traditional relational database as if it were a high-performance vector retrieval engine.

## 4.8. Data Collection Conclusion

The data collection approach in the project is therefore strict, privacy-aware, and structurally simple. It defines identity-based inputs, quality gates, clear storage rules, and a research narrative that accommodates both a pretrained baseline and a custom branch. This gives the diploma a coherent basis for discussing both technical and governance aspects of biometric data handling.

[[PAGE_BREAK]]
# 5. Methodology of the Work

## 5.1. General Methodological Approach

The methodology of the work combines architectural decomposition, algorithmic face processing, vector search, and operator-centered interface design. The project is not treated as a single neural network problem but as a system of interacting stages. A face search request is meaningful only if the client can submit an image, the backend can validate it, the detector can locate faces, the embedder can transform them into vectors, the search layer can retrieve candidates quickly, and the operator can review the result. This end-to-end view guides the methodology of the project.

At a technical level, the work uses the following chain:

1. input acquisition;
2. image decoding and validation;
3. face detection;
4. optional alignment and quality control;
5. embedding extraction;
6. approximate nearest-neighbor vector search;
7. metadata enrichment and ranking;
8. operator-facing output.

Each stage is implemented as a logical subsystem rather than being merged into one opaque function. This improves maintainability and makes the system suitable for architecture analysis.

## 5.2. Enrollment Method

Enrollment is the process of adding a new identity sample into the biometric repository. In this project it is intentionally conservative. The backend receives one image and an optional label. The image is decoded and checked. If no face is found, the request fails with a validation response. If multiple faces are found, the request also fails. Only when exactly one face is detected does the backend proceed to embedding extraction.

The resulting vector is stored together with person metadata and then inserted into the corresponding FAISS index. Because the system supports multiple logical pipelines, enrollment can target the pretrained pipeline, the custom pipeline, or both. This makes enrollment not only a data-entry mechanism but also a comparative experiment mechanism.

## 5.3. Search Method

Search is formulated as a similarity retrieval problem. The operator submits an image or the desktop sends a webcam frame. The backend decodes the input, detects all faces present, and processes each detected face independently. For each face, an embedding is created and sent to the selected FAISS index. The system retrieves the top-k nearest vectors and then enriches those results with database metadata such as person label and identifier.

The search response therefore contains both numerical and semantic layers:

- score and distance values for ranking;
- person identifiers and labels for operator interpretation;
- per-face grouping when multiple faces are detected;
- decision logic such as match versus unknown according to thresholding.

## 5.4. Compare Mode Method

Compare mode is one of the most important methodological elements of the project. Instead of treating the system as bound to one model, compare mode runs two logical pipelines on the same input:

- a pretrained baseline pipeline;
- a custom model pipeline.

The purpose of this mode is not only operational convenience. It creates a controlled environment in which the same image, the same UI, and the same approximate search context can be used to compare model behavior, latency, and output ranking.

## 5.5. Multiple-Face Search

Multiple-face search is methodologically justified by real usage scenarios. Group images, archive photos, or webcam frames may contain more than one person. The system therefore processes each detected face separately during search. Each face receives its own search result set, even if the operator submitted only one image. This capability is especially relevant for the desktop live mode, where a frame may show one, two, or more people at once.

The key methodological distinction remains: multi-face input is valid for search, but not for enrollment. This preserves repository cleanliness while supporting realistic operational use.

## 5.6. Live Webcam Workflow

The desktop includes a live webcam-assisted mode. The methodological decision here is to implement near real-time scanning rather than full video-rate inference on every single frame. The desktop shows a local preview, periodically selects a frame, sends it to the backend, and displays the returned search or compare result. This approach is more practical for a student project because it balances responsiveness with backend cost and avoids turning the system into a streaming video analytics platform.

## 5.7. Evaluation Methodology

The project evaluates practical behavior along several axes:

- runtime responsiveness;
- operator usability;
- support for enrollment and search workflows;
- ability to rebuild and inspect indices;
- ability to compare model branches;
- compliance with the “no raw image storage by default” rule.

From an algorithmic perspective, the project distinguishes between synthetic retrieval metrics and biometric verification metrics. The synthetic benchmark reports latency percentiles, build time, serialized index size, and `top_k_overlap@K` against exact Flat search. Biometric verification requires labeled positive and negative pairs and is described through FAR, FRR, EER, TAR@FAR, and threshold behavior. From a system perspective, equally important metrics are the ability to start locally or via Docker, the clarity of API contracts, and the quality of the operator interface.

## 5.8. Methodology Conclusion

The methodology of the work is therefore system-centered and comparative. It unites face-processing algorithms, vector indexing, strict data rules, and operator workflows in a single coherent design. This methodology is appropriate for the diploma because it demonstrates both engineering discipline and applied research value.

[[PAGE_BREAK]]
# 6. MVP / UML / Architecture of the Project

## 6.1. Architecture Design Principles

The architecture follows a modular approach inside a clear service boundary. The core idea is to keep logical responsibilities separated even when the prototype is deployed as a compact system. The main internal boundaries are API controllers and request schemas, face-processing services, embedding extractor interfaces and adapters, vector index interfaces and FAISS implementations, repository layer for metadata and embeddings, and a desktop client as a separate operator-facing application.

This design reflects the architectural decisions documented in the supporting architecture file `SAFinal.docx`. The rationale is that a prototype should remain easy to deploy while still being structured enough to evolve into a larger system if required.

## 6.2. System Context

The system context diagram shows the main actors and the top-level software boundary.

![Figure 1. System context of the biometric face search project.](assets/safinal/figure_01_system_context.png)

Figure 1 illustrates that the operator and administrator do not interact directly with raw model services or the database. Instead, they work through a client application that communicates with the biometric search system. The operator performs search and enrollment activities, while the administrator performs maintenance tasks such as index rebuild.

## 6.3. Container View

The container-level diagram explains how runtime responsibilities are divided between the application layer, ML runtime, index management, and relational storage.

![Figure 2. Container view of the deployed prototype.](assets/safinal/figure_02_container_view.png)

The most important architectural point in Figure 2 is the separation of metadata storage from vector search. PostgreSQL is used for people, embeddings, and indexing status information, while the FAISS active index is responsible for nearest-neighbor search. The ML runtime is a dedicated logical block for detection, quality checking, alignment, and embedding extraction.

## 6.4. Component View

The component diagram gives a more detailed view inside the API service boundary.

![Figure 3. Component-level view of the API service.](assets/safinal/figure_03_component_view.png)

This diagram is particularly useful for the diploma because it shows how the system is decomposed into controller, service, adapter, and repository responsibilities. Search and enrollment are represented explicitly as services rather than as ad hoc controller logic. The FAISS adapter is separated from the repository layer, which makes the architecture easier to explain and test.

## 6.5. BPMN View of Enrollment

The enrollment business process is shown below.

![Figure 4. BPMN view of the enrollment process.](assets/safinal/figure_04_enroll_bpmn.png)

The BPMN diagram makes visible the decision points that matter at the business-process level: whether the request is valid, whether face quality is acceptable, whether the image contains exactly one face, whether embedding extraction succeeds, and whether indexing occurs synchronously or asynchronously.

## 6.6. BPMN View of Search

The search business process is also represented as a workflow.

![Figure 5. BPMN view of the search process.](assets/safinal/figure_05_search_bpmn.png)

This diagram emphasizes that search is not merely a top-k lookup. The system validates the request, detects faces, checks whether the active index exists, applies thresholding and ranking, and fetches metadata for candidates. In a practical biometric system these steps matter because the operator needs interpretable outcomes, not only distances in vector space.

## 6.7. Sequence Diagram for Search

The sequence view of `/search` clarifies the latency-critical path.

![Figure 6. Sequence diagram for search.](assets/safinal/figure_06_search_sequence.png)

Figure 6 shows the path from client request to authentication, face processing, approximate nearest-neighbor lookup, metadata retrieval, audit logging, and response. The diagram is important because it explains which layers are on the synchronous critical path and therefore influence operator-perceived latency.

## 6.8. Sequence Diagram for Enrollment

The enrollment sequence view complements the search sequence.

![Figure 7. Sequence diagram for enroll.](assets/safinal/figure_07_enroll_sequence.png)

This sequence diagram shows that enrollment involves both storage and index update decisions. If vector insertion succeeds immediately, the system can reflect the new person in the live search path without a full rebuild. If not, the system can mark indexing as pending and rebuild later. This is a practical compromise between operational simplicity and index consistency.

## 6.9. Architectural Decisions and Trade-Offs

The architectural supporting material in `SAFinal.docx` identifies several key decisions that directly shaped the final implementation:

- modular pipeline inside a single deployable service boundary;
- deep embeddings rather than handcrafted descriptors;
- FAISS as the primary vector search engine;
- storage of embeddings and metadata without raw image retention by default;
- synchronous REST for operator flows and more controlled handling for heavier maintenance actions;
- containerized, single-node-first deployment.

These choices create clear trade-offs. A single-node architecture simplifies deployment and pre-defense demonstration but offers less independent scaling than a full microservice layout. FAISS gives excellent control for experiments but requires more explicit persistence and snapshot logic than a fully managed vector platform. Avoiding raw image storage reduces privacy risk but means re-embedding requires access to source datasets when model changes are significant.

## 6.10. Runtime Pipelines in the Final Project

The current project architecture supports two logical runtime pipelines:

- **Pretrained pipeline**: the stable operational baseline, implemented through ONNX runtime assets and used as the main working reference for search and comparison.
- **Custom pipeline**: an experimental PyTorch branch intended for comparative research; final quality claims require valid training logs and labeled verification results.

The dual-pipeline architecture is pedagogically useful because it allows the diploma to present not only a finished baseline but also a comparative methodology. In other words, the system is not limited to one model that happens to work; it is structured to compare branches, explain trade-offs, and support future evolution.

## 6.11. Architecture Conclusion

The architectural design of the project is one of its strongest aspects. It combines practical deployment simplicity with clear boundaries, explicit diagrams, and justified technology decisions. This makes the system suitable both for implementation and for academic defense.

[[PAGE_BREAK]]
# 7. Technology Comparison

## 7.1. Backend Framework: FastAPI

FastAPI was selected as the backend framework because it provides a clear and productive way to define REST endpoints, request validation, and typed application structure. Alternatives such as Flask are lighter but require more manual assembly for schema-driven APIs. Full-stack frameworks such as Django would introduce more built-in machinery than the project needs for a focused biometric service. FastAPI therefore provides a strong balance between speed of implementation, type clarity, and modern async-compatible architecture [18].

## 7.2. Storage: PostgreSQL and SQLite

The project supports PostgreSQL as the main containerized database and SQLite as a convenient local development mode. PostgreSQL is a better fit for realistic deployment because it offers stronger concurrent behavior, richer data types, and clearer migration management. SQLite remains useful for rapid local experimentation and single-user runs. Supporting both does not weaken the architecture; it makes the prototype easier to demonstrate while still preserving a more serious database path for deployment.

## 7.3. Vector Search: FAISS Versus Database-Only Search

One of the central technology choices is to perform vector search in FAISS rather than directly in the relational database. A database-only approach would be simpler conceptually but far less suitable for large-scale nearest-neighbor retrieval. FAISS supports exact and approximate vector search with configurable algorithms such as flat search, HNSW, and IVF-PQ. This is both a performance decision and a research decision. It allows the project to discuss approximate nearest-neighbor trade-offs explicitly rather than hiding them inside a generic database query engine [5], [13], [14].

## 7.4. Runtime Inference: ONNX and PyTorch

The pretrained baseline is implemented through ONNX runtime assets, while the custom research line is associated with the project’s own model branch. ONNX is attractive for operational inference because it is lightweight, portable, and stable once the model is exported. PyTorch is attractive for training, experimentation, and model customization. This difference reflects a common industry pattern: training-oriented flexibility on one side, deployment-oriented portability on the other.

## 7.5. Desktop Interface: PySide6

PySide6 was chosen for the desktop client because it supports native-feeling widget applications, structured UI composition, and access to local resources such as file dialogs and webcam devices. A web UI would also have been possible, but the desktop format better matches operator workflows in controlled workstation environments. It also supports an integrated live webcam mode without introducing a browser as an additional moving part.

## 7.6. Deployment: Docker Compose

Docker Compose was selected for reproducible local deployment because it is simple, understandable, and adequate for a diploma prototype. Orchestrators such as Kubernetes would provide stronger scalability and isolation features, but they would also introduce unnecessary operational overhead for the current scope. The project needs to demonstrate a working reproducible system, not a full enterprise distributed cluster.

## 7.7. Technology Comparison Conclusion

The chosen stack is therefore coherent:

- FastAPI provides clear service-level APIs;
- PostgreSQL and SQLite cover deployment and local development;
- FAISS provides explicit vector-search control;
- ONNX and PyTorch support baseline and research model lines;
- PySide6 provides a strong desktop operator interface;
- Docker Compose offers reproducible deployment without excessive operational complexity.

This combination is well aligned with the goals of a fast-search biometric diploma project.

[[PAGE_BREAK]]
# 8. Implemented Desktop Interface Screens

## 8.1. Dashboard

The dashboard presents a high-level overview of backend health, active pipelines, number of profiles, and recent activity. It is designed to orient the operator quickly and to show whether the system is in a working state before search operations begin.

![Figure 8. Desktop dashboard screen.](assets/screenshots/desktop_dashboard.png)

The dashboard is intentionally concise: it emphasizes pipeline availability, index state, and quick navigation rather than overwhelming the operator with low-level details. This makes it appropriate for the beginning of an operator workflow.

## 8.2. Face Search Screen

The main search screen is the central operational page of the desktop client. It supports image-based search, enrollment, compare mode, and live camera interaction in one place.

![Figure 9. Face Search screen with result workflow.](assets/screenshots/desktop_search.png)

This screen is particularly important for the diploma because it demonstrates that the project is not just an API. It is a usable application with a structured operator flow, input handling, result cards, and status indicators.

## 8.3. Database Screen

The database page displays person records and embedding-related metadata. It gives the operator or reviewer a way to inspect the contents of the repository rather than treating it as an invisible backend detail.

![Figure 10. Database screen with profile cards and details.](assets/screenshots/desktop_database.png)

The presence of this page strengthens the practical value of the project. A biometric system that cannot inspect stored records is significantly less useful in real operational contexts.

## 8.4. Logs and Statistics Screen

The logs and statistics page is responsible for backend status inspection, recent event review, and index maintenance actions.

![Figure 11. Logs and statistics screen.](assets/screenshots/desktop_logs.png)

This page is valuable for both debugging and demonstration. It makes the system feel operationally complete and supports academic discussion of observability and index management.

## 8.5. Interface Conclusion

The implemented desktop interface screens show that the project includes a coherent interaction layer rather than a collection of disconnected demo calls. This is important for the diploma because it demonstrates a full software product perspective.

[[PAGE_BREAK]]
# 9. Experimental and Practical Results

## 9.1. Working Baseline

The project currently operates with a stable pretrained baseline pipeline that serves as the most reliable runtime path for demonstration and comparative analysis. This branch is used as the operational anchor of the system. It supports enrollment, search, index maintenance, desktop presentation, and live interaction with the backend.

From a practical standpoint, the existence of a stable baseline is essential. It allows the project to be evaluated not only as a conceptual design but as a functioning system. The desktop client, the REST API, the FAISS index, and the storage model all interact successfully through this baseline.

## 9.2. Custom Model Branch

The project also includes a custom PyTorch model branch. This branch is best described as an experimental research extension rather than as a validated replacement for the stable pretrained path. It contains training, evaluation, and export tooling, but final claims about custom-model biometric accuracy require complete training evidence and labeled verification results.

From an academic perspective, this is still useful. The branch demonstrates that the architecture can support project-specific models, while the stable MVP remains anchored in the pretrained extractor path for reproducible demonstration.

## 9.3. Dual-Pipeline Compare Mode

One of the most practically valuable features implemented in the system is compare mode. It allows the same image to be processed by both model branches and displays their outputs in a unified operator interface. This is useful in three ways:

- it demonstrates the comparative logic of the diploma;
- it provides transparency for model behavior;
- it enables practical benchmarking and review of latency and ranking differences.

## 9.4. Multiple-Face Search and Live Webcam Mode

The system supports multiple-face search, which is important for scene images and live frames. In addition, the desktop implements a live webcam mode that periodically sends frames to the backend and visualizes results in the interface. This mode is near real-time rather than full frame-rate analytics, but it is sufficient for operator demonstration and controlled evaluation.

## 9.5. Practical Strengths of the Current Prototype

The main practical strengths of the system are:

- clear separation between storage and vector search;
- explicit handling of enrollment versus search rules;
- support for both image-based and live inputs;
- dual runtime comparison;
- desktop interface with database and logs views;
- reproducible containerized deployment path.

## 9.6. Current Limitations

The project also has limitations that should be stated honestly:

- the pretrained branch is currently the most stable working runtime path;
- the custom PyTorch branch remains experimental and requires valid training logs plus labeled verification evaluation before model-quality claims can be made;
- live webcam mode is near real-time and not intended as a high-FPS surveillance platform;
- final stable-extractor biometric accuracy studies and threshold calibration require labeled pair evaluation.

## 9.7. Practical Application Areas

Despite the prototype status, the project already demonstrates relevance to several application areas:

- operator-assisted identity search in controlled facilities;
- archive and registry search by face image;
- assisted access control workflows;
- comparative study of biometric search pipelines;
- educational demonstration of vector-search-based recognition systems.

## 9.8. Results Conclusion

The main result of the work is not a single accuracy number. It is the successful creation of a complete biometric face search prototype that unites model inference, vector search, storage, desktop interaction, and comparative runtime logic in one architecture. This is the central contribution of the project.

The scientific and engineering contribution must be separated from external algorithms. The project does not claim to invent ArcFace, SCRFD, HNSW, IVF-PQ, or a new biometric loss. Those are external methods and libraries used as components. The author's contribution is the integrated system: a modular FastAPI backend, desktop operator client, custom Torch pipeline integration, storage and encryption logic for biometric templates, FAISS index management, real LFW verification reporting, synthetic scalability benchmarking, and clear claim boundaries for biometric quality versus retrieval behavior.

[[PAGE_BREAK]]
# 10. Security and Privacy Considerations

Biometric systems require a stricter security and privacy discussion than ordinary CRUD applications because biometric templates cannot be rotated as easily as passwords. The project therefore applies several MVP-level safeguards and states the remaining limitations explicitly.

Raw input face images are not stored by default. Images are used to extract face embeddings and then discarded by the backend request path. The persistent data model stores person metadata, encrypted embedding payloads, encrypted FAISS snapshot files, snapshot metadata, and audit logs. Embeddings and FAISS index snapshots remain sensitive biometric template artifacts even when raw images are not retained.

API-key handling was hardened so that key comparison uses timing-safe comparison. Example environment files use placeholders rather than reusable real-looking keys, and fresh `API_KEY`, `ADMIN_API_KEY`, `DATA_ENCRYPTION_KEY`, and `SNAPSHOT_ENCRYPTION_KEY` values must be generated before using real biometric data. The rate limiter protects search, compare, enroll, rebuild, and delete routes in the local MVP. Its bucket identity uses the API-key digest when an API key is available, or client IP otherwise; raw API keys are not logged for rate-limit buckets.

The project also includes snapshot retention. `INDEX_SNAPSHOT_RETENTION` limits the number of retained FAISS snapshots per index path, and matching `.map.json` sidecars are pruned with old snapshots. This reduces disk growth and limits retention of old biometric index artifacts, but it does not remove the need to protect snapshot storage, backups, and filesystem access.

The security posture should be described as MVP hardening, not as a complete enterprise security program. The in-memory rate limiter is per-process and not distributed. A deployment with multiple workers or multiple hosts would require Redis, an API gateway, WAF-level controls, or another external limiter. HTTPS termination, secret rotation, backup encryption, network policy, and operational monitoring are deployment responsibilities. Full RBAC is not implemented. A hard purge endpoint for irreversible deletion of all person-related biometric templates and index artifacts is not implemented yet. Liveness detection and spoofing resistance are also future work.

[[PAGE_BREAK]]
# 11. Custom Model Status

The custom PyTorch branch is the proposed custom runtime pipeline of the project. It demonstrates that the architecture can host project-specific embedding extractors, training utilities, candidate checkpoints, and export workflows. The branch is now supported by labeled LFW verification results, so it can be discussed quantitatively rather than only methodologically.

The final custom runtime uses `torch_insightface_iresnet100` with `runtime_fallback_center_crop` preprocessing, RGB input, `[-1, 1]` normalization, hflip TTA, and selected threshold 0.205047. On LFW, this final custom runtime achieved EER 0.015000 and best accuracy 0.990500 on 6000/6000 valid pairs. The pretrained ONNX/InsightFace baseline achieved EER 0.027852 and best accuracy 0.984556 on its evaluated valid pairs. The correct conclusion is that the custom runtime is implemented, evaluated with real biometric metrics, and compared against an external baseline. This does not imply liveness protection, universal deployment accuracy, or compatibility with old `torch_ir50` embeddings.

[[PAGE_BREAK]]
# 12. Defense Claim Boundaries

The following claim boundaries keep the defense narrative aligned with the implementation:

- It is correct to say that FAISS-based vector retrieval is implemented and benchmarked on synthetic embeddings.
- It is correct to say that Flat, HNSW, and IVF-PQ are available as index methods in the research benchmark.
- It is correct to say that `top_k_overlap@K` measures overlap with exact Flat top-K neighbors in the synthetic benchmark.
- It is not correct to call `top_k_overlap@K` biometric identification hit@K.
- It is correct to report the tracked LFW FAR/FRR/EER/TAR@FAR results for the custom Torch IR-50 pipeline and the pretrained ONNX/InsightFace baseline.
- It is not correct to claim that the custom pipeline is better than the pretrained baseline.
- It is correct to say that the stable extractor pair evaluator exists and can evaluate ONNX, InsightFace, Torch, or Dummy backends.
- It is not correct to treat the Dummy backend as biometric evidence.
- It is correct to say that raw images are not stored by default and that embeddings and snapshots are encrypted.
- It is correct to say that API-key comparison is timing-safe and that configurable in-memory rate limiting exists.
- It is not correct to claim complete enterprise security, full DDoS protection, full RBAC, liveness detection, or hard purge compliance.

[[PAGE_BREAK]]
# 13. Experimental Results and Evaluation Methodology

## 13.1. Evaluation Layers

The current evaluation plan is divided into three layers:

- synthetic vector retrieval benchmark;
- biometric verification threshold methodology;
- stable extractor pair evaluator readiness.

This separation is important. Vector retrieval evaluation measures FAISS index behavior after embeddings already exist. Biometric verification evaluation measures whether similarity scores separate same-identity and different-identity pairs. Runtime demonstration shows that API, desktop, storage, and indexing workflows operate together, but it is not a substitute for labeled biometric accuracy evaluation.

## 13.2. Synthetic Retrieval Benchmark Methodology

The synthetic retrieval benchmark uses L2-normalized 512-dimensional embeddings generated with a fixed NumPy seed. Flat exact search is used as the baseline. HNSW and IVF-PQ are evaluated as approximate indexes. Latency is measured as repeated single-query `index.search` calls after warmup. Build time includes index construction and vector insertion; IVF-PQ build time also includes training. Memory estimate is computed as serialized FAISS index size, not full process RSS.

The retrieval metric is:

`top_k_overlap@K = |exact_top_K(query) ∩ approximate_top_K(query)| / K`

This is not biometric identification hit@K and not biometric accuracy. The benchmark has no identity labels and therefore cannot measure whether a person was correctly identified. It only measures how closely an approximate index reproduces exact Flat nearest-neighbor retrieval on synthetic vectors.

## 13.3. Synthetic Retrieval Benchmark Results

The following values are copied from `docs/benchmarks/retrieval_benchmark_pr2.md`. No values were changed or estimated.

| Database size | Embedding dim | Queries | Seed | Method | Index parameters | Build time (s) | Memory estimate (MB) | p50 latency (ms) | p95 latency (ms) | p99 latency (ms) | top_k_overlap@1 | top_k_overlap@5 | top_k_overlap@10 |
|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 512 | 100 | 42 | flat | `{"metric":"inner_product","normalized_vectors":true}` | 0.000081 | 0.195355 | 0.004500 | 0.008610 | 0.021691 | 1.000000 | 1.000000 | 1.000000 |
| 100 | 512 | 100 | 42 | hnsw_ef64 | `{"M":32,"efConstruction":200,"efSearch":64,"metric":"inner_product","normalized_vectors":true}` | 0.001760 | 0.221331 | 0.022700 | 0.024515 | 0.034254 | 1.000000 | 1.000000 | 1.000000 |
| 100 | 512 | 100 | 42 | ivfpq_nprobe8 | `{"M_pq":16,"metric":"inner_product","nbits":4,"nlist":16,"normalized_vectors":true,"nprobe":8}` | 0.019342 | 0.064320 | 0.008500 | 0.009210 | 0.009929 | 1.000000 | 0.290000 | 0.401000 |
| 1000 | 512 | 100 | 42 | flat | `{"metric":"inner_product","normalized_vectors":true}` | 0.000552 | 1.953168 | 0.037450 | 0.052320 | 0.059548 | 1.000000 | 1.000000 | 1.000000 |
| 1000 | 512 | 100 | 42 | hnsw_ef64 | `{"M":32,"efConstruction":200,"efSearch":64,"metric":"inner_product","normalized_vectors":true}` | 0.013336 | 2.212221 | 0.061900 | 0.070200 | 0.072136 | 1.000000 | 0.994000 | 0.997000 |
| 1000 | 512 | 100 | 42 | ivfpq_nprobe8 | `{"M_pq":16,"metric":"inner_product","nbits":4,"nlist":16,"normalized_vectors":true,"nprobe":8}` | 0.070546 | 0.078053 | 0.019800 | 0.020505 | 0.021358 | 0.130000 | 0.056000 | 0.052000 |
| 10000 | 512 | 100 | 42 | flat | `{"metric":"inner_product","normalized_vectors":true}` | 0.005819 | 19.531293 | 0.616750 | 0.976610 | 1.045482 | 1.000000 | 1.000000 | 1.000000 |
| 10000 | 512 | 100 | 42 | hnsw_ef64 | `{"M":32,"efConstruction":200,"efSearch":64,"metric":"inner_product","normalized_vectors":true}` | 1.786023 | 22.123201 | 0.333450 | 0.460055 | 0.472427 | 1.000000 | 0.994000 | 0.987000 |
| 10000 | 512 | 100 | 42 | ivfpq_nprobe8 | `{"M_pq":16,"metric":"inner_product","nbits":4,"nlist":16,"normalized_vectors":true,"nprobe":8}` | 0.174341 | 0.215382 | 0.131150 | 0.148080 | 0.167981 | 0.030000 | 0.012000 | 0.009000 |

The benchmark environment was Windows 11, Python 3.12.10, NumPy 2.4.2, FAISS 1.13.2, 20 CPU threads, 512-dimensional embeddings, 100 queries per database size, seed 42, and query seed 123. A 100,000-vector result is not reported because that run was not executed for these tracked artifacts.

## 13.4. Interpretation

Flat is the exact baseline used to compute reference nearest-neighbor lists. Flat may be faster on very small datasets because approximate-index overhead is not justified. As database size grows, HNSW becomes useful because it preserves high top-k overlap while reducing search latency in the 10,000-vector run. IVF-PQ reduces serialized index size in the current configuration, but the measured top-k overlap is low on these synthetic vectors. Therefore, the IVF-PQ configuration should be presented as a memory-saving experimental setting rather than as a quality-equivalent replacement for exact search.

The synthetic retrieval benchmark evaluates vector retrieval behavior, not biometric recognition quality.

## 13.5. Verification Metrics and Threshold Calibration

The project includes pure NumPy helpers for biometric verification metrics:

- FAR: false accepts divided by all negative pairs;
- FRR: false rejects divided by all positive pairs;
- EER: the point where FAR and FRR are closest or safely interpolated;
- TAR@FAR: true accept rate at a selected target FAR;
- best accuracy threshold;
- FAR/FRR curve points.

The score convention is explicit: higher score means more similar, and `score >= threshold` means match. Raising the threshold lowers FAR and increases FRR. Lowering the threshold increases FAR and lowers FRR. This threshold behavior must be calibrated on labeled positive and negative pairs.

The final LFW labeled-pair evaluation was run for the final custom Torch runtime model and for the pretrained ONNX/InsightFace baseline. The final custom `torch_insightface_iresnet100` pipeline achieved EER 0.015000, best accuracy 0.990500, and TAR@FAR=0.01 equal to 0.984667 on 6000/6000 valid pairs with zero skipped pairs. The selected runtime threshold is 0.205047, with FAR 0.001667 and FRR 0.017333 at that threshold. The pretrained baseline achieved EER 0.027852, best accuracy 0.984556, and TAR@FAR=0.01 equal to 0.971141 on 5957 valid pairs. These values are real biometric verification metrics and must be interpreted separately from synthetic FAISS retrieval metrics.

## 13.6. Stable Extractor Evaluator

The standalone evaluator `scripts/evaluate_lfw_verification.py` reads LFW-style `pairs.txt`, loads image pairs, extracts embeddings through either the custom Torch pipeline or the pretrained ONNX/InsightFace baseline, L2-normalizes embeddings, computes dot-product similarity, and reports FAR, FRR, EER, TAR@FAR, selected thresholds, and curve points.

The final evaluation uses `handoff_lfw_eval/lfw` and `handoff_lfw_eval/lfw/pairs.txt`. LFW is reserved for evaluation and threshold analysis, not for final training. The main real-face fine-tuning dataset is defined as `datasets/celeba_faces/train` with validation data in `datasets/celeba_faces/val`.

## 13.7. Limitations of Evaluation

The current evaluation state has clear limitations:

- LFW metrics are now reported for the final custom runtime model, but they remain an evaluation result rather than a guarantee for every operating condition;
- the synthetic retrieval benchmark is not biometric accuracy;
- historical `torch_ir50` embeddings remain incompatible with the final `torch_insightface_iresnet100` embedding space and must be re-enrolled or re-imported from source images;
- a 100,000-vector benchmark is not reported because it was not run for the tracked PR 2 artifacts;
- liveness, spoofing resistance, and operational abuse testing remain future work.

[[PAGE_BREAK]]
# 14. Conclusion

This diploma project addressed the problem of fast search in biometric databases through the design and implementation of a modular face-based identification system. The work demonstrated that practical biometric search should be understood as a full-stack system problem involving detection, embedding extraction, vector search, metadata persistence, operator interaction, and deployment discipline.

The project achieved the main goals formulated at the start of the work. It implemented a FastAPI backend, a FAISS-based search layer, relational metadata storage, a PySide6 desktop client, and a dual-pipeline runtime design. The system supports strict enrollment, multi-face search, comparison of model branches, index maintenance, and near real-time webcam-assisted search. The design is supported by architecture diagrams, process diagrams, sequence diagrams, and implemented interface screenshots.

The diploma also formulated a clear architectural position: images should not be stored by default, vector search should be delegated to FAISS rather than to a relational database, and runtime modularity should be preserved even in a compact deployment. These decisions make the system more understandable, more privacy-aware, and more extensible.

An important conceptual outcome of the work is the separation between a stable pretrained runtime path and an experimental custom PyTorch research branch. This framing gives the project a meaningful comparative dimension while avoiding unsupported model-quality claims.

Future work may include larger-scale benchmarking, labeled stable-extractor threshold calibration, hard purge support, fuller RBAC, liveness or spoofing checks, and improved operator analytics. Nevertheless, even in its current form the project represents a solid and defensible software engineering result: a complete, modular, and practical biometric face search system suitable for diploma defense discussion.

[[PAGE_BREAK]]
# 15. References

1. Jain, A. K., Ross, A., and Prabhakar, S. “An Introduction to Biometric Recognition.” IEEE Transactions on Circuits and Systems for Video Technology, 2004.
2. Li, S. Z., and Jain, A. K. *Handbook of Face Recognition*. Springer, 2011.
3. Bowyer, K. W., Chang, K., and Flynn, P. “A Survey of Approaches and Challenges in 3D and Multi-Modal 3D+2D Face Recognition.” Computer Vision and Image Understanding, 2006.
4. Deng, J., Guo, J., Zhou, Y., Yu, J., Kotsia, I., and Zafeiriou, S. “RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild.” 2020.
5. Johnson, J., Douze, M., and Jégou, H. “Billion-Scale Similarity Search with GPUs.” IEEE Transactions on Big Data, 2019.
6. Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., and Song, L. “SphereFace: Deep Hypersphere Embedding for Face Recognition.” CVPR, 2017.
7. Phillips, P. J., et al. “An Introduction to the Good, the Bad, & the Ugly Face Recognition Challenge Problem.” FG, 2011.
8. Turk, M., and Pentland, A. “Eigenfaces for Recognition.” Journal of Cognitive Neuroscience, 1991.
9. Ahonen, T., Hadid, A., and Pietikäinen, M. “Face Description with Local Binary Patterns.” ECCV, 2004.
10. Schroff, F., Kalenichenko, D., and Philbin, J. “FaceNet: A Unified Embedding for Face Recognition and Clustering.” CVPR, 2015.
11. Deng, J., Guo, J., Xue, N., and Zafeiriou, S. “ArcFace: Additive Angular Margin Loss for Deep Face Recognition.” CVPR, 2019.
12. Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. “Joint Face Detection and Alignment Using Multi-Task Cascaded Convolutional Networks.” IEEE Signal Processing Letters, 2016.
13. Malkov, Y. A., and Yashunin, D. A. “Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.” IEEE TPAMI, 2020.
14. Jégou, H., Douze, M., and Schmid, C. “Product Quantization for Nearest Neighbor Search.” IEEE TPAMI, 2011.
15. Nielsen, J. *Usability Engineering*. Morgan Kaufmann, 1994.
16. ISO/IEC 24745. *Information Technology — Security Techniques — Biometric Information Protection*.
17. Cavoukian, A. *Privacy by Design: The 7 Foundational Principles*. Information and Privacy Commissioner of Ontario, 2011.
18. Ramírez, S. *FastAPI Documentation*. FastAPI official documentation.
