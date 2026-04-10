from dataclasses import dataclass, fields as dc_fields
from typing import Tuple
"""
---------------------------------------------------------------------------
Parameter dataclasses
---------------------------------------------------------------------------

Each parameter is a 3-tuple: (value_str, type_str, unit_str)
type_str: "string" | "double" | "bool"
unit_str: "" | "m" | "mm" | "rad"
"""
Param = Tuple[str, str, str]

class _StepParams:
    """Mixin: convert dataclass fields to the params_dict format for set_params()."""

    def to_step_params(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self) if f.name != "name"}

@dataclass
class EasyCreateStringList(_StepParams):
    """Parameters for the Easy Create String List vision step.
    Used to inject the scene PLY file path into the Calc Result by Python step.
    """
    name:       str   = "Scene_Path"
    strings:    Param = ("", "string", "")   # absolute path to merged scene PLY lists


@dataclass
class CoarseMatchingV2(_StepParams):
    """Parameters for the 3D Coarse Matching V2 vision step.

    Ref: https://docs.mech-mind.net/en/suite-software-manual/1.8.2/vision-steps/3d-coarse-matching-v2.html

    Model:
        name:                           MechVision step name in the project workflow.
        modelSelection:                 Name of the model in the model library
                                        (project/resource/3d_matching/<name>).
        modelFileName:                  Absolute path to the PLY point cloud model file.
        geoCenterFileName:              Absolute path to the JSON geometric center file.

    Matching method:
        registrationMode:               Matching algorithm. 0.0 = Surface matching (objects
                                        with recognizable surfaces), 1.0 = Edge matching
                                        (flat objects with clear edges).

    Preprocessing:
        autoCalculateExpectedPointsNum: When True, automatically calculates the expected
                                        point count of the down-sampled model point cloud.
                                        Recommended to leave enabled.
        maxScenePointNum:               Upper threshold for the sampled input point cloud.
                                        If the scene point count exceeds this value the
                                        step outputs nothing. Default: 1,000,000.

    Voting:
        maxNumOfPointPairsPerFeature:   Upper limit of point pairs sampled per feature.
                                        Lower values are faster but less accurate.
                                        Default: 50.
        distQuantification:             Distance quantification factor.
                                        DistanceInterval = distQuantification × SamplingInterval.
                                        Higher values reduce accuracy. Default: 1.0.
        angleQuantification:            Angle quantification factor.
                                        AngleInterval = 360° / angleQuantification.
                                        Higher values increase accuracy but require higher
                                        quality input data. Default: 60.
        maxVoteRatio:                   Vote ratio lower threshold. Only candidate poses
                                        scoring above (HighestScore × maxVoteRatio) proceed
                                        to verification. Lower values increase computation
                                        time. Default: 0.80.
        refStep:                        Referring point sampling step. Larger step sizes
                                        speed up execution but reduce accuracy. Default: 5.
        referredStep:                   Referred point sampling step.  Larger step sizes
                                        speed up execution but reduce accuracy. Default: 1.

    Pose filtering:
        useDistanceNMS:                 When True, removes duplicate poses whose distance
                                        to an already-selected pose is less than 0.1× the
                                        model diameter. Default: True.
        filterCandidatePoseByAxis:      (Edge matching only) Filter candidate poses that
                                        exceed the rotation angle threshold. Default: True.
        angleThreshold:                 (Edge matching only) Maximum allowed rotation angle
                                        difference in degrees. Applied when
                                        filterCandidatePoseByAxis is True. Default: 135°.

    Pose verification:
        voxelLengthGenetationStrategy:  Strategy for setting voxel grid size used in pose
                                        verification. 0.0 = Auto (recommended),
                                        1.0 = Manual (uses minVoxelLength / maxVoxelLength).
                                        Note: key name typo is intentional — matches the
                                        MechVision gRPC key.
        minVoxelLength:                 (Manual strategy only) Lower limit of the voxel
                                        edge length in mm. Default: 1.0 mm.
        maxVoxelLength:                 (Manual strategy only) Upper limit of the voxel
                                        edge length in mm. Default: 15.0 mm.
        outputNum:                      Expected number of detected poses per input point
                                        cloud. Default: 3.
    """

    name:                           str   = "Coarse_Match_Synthetics"
    modelSelection:                 Param = ("", "string", "")
    modelFileName:                  Param = ("", "string", "")
    geoCenterFileName:              Param = ("", "string", "")
    registrationMode:               Param = ("0.0", "double", "")       # 0.0=Surface, 1.0=Edge

    # Preprocessing
    autoCalculateExpectedPointsNum: Param = ("True", "bool", "")        # constant True is ok, seemed to be robust
    maxScenePointNum:               Param = ("1000000", "double", "")

    # Voting
    maxNumOfPointPairsPerFeature:   Param = ("10000", "double", "")
    distQuantification:             Param = ("1.0", "double", "")
    angleQuantification:            Param = ("60", "double", "")
    maxVoteRatio:                   Param = ("0.5", "double", "")
    refStep:                        Param = ("5", "double", "")
    referredStep:                   Param = ("1", "double", "")

    # Pose filtering
    useDistanceNMS:                 Param = ("True", "bool", "")
    filterCandidatePoseByAxis:      Param = ("True", "bool", "")        # Edge only
    angleThreshold:                 Param = ("0", "double", "")         # Edge only

    # Pose verification
    voxelLengthGenetationStrategy:  Param = ("0.0", "double", "")       # 0.0=Auto (typo is intentional — matches MechVision key)
    minVoxelLength:                 Param = ("0.001", "double", "mm")
    maxVoxelLength:                 Param = ("0.001", "double", "mm")
    outputNum:                      Param = ("1", "double", "")


@dataclass
class FineMatchingLite(_StepParams):
    """Parameters for the 3D Fine Matching Lite vision step.

    Ref: https://docs.mech-mind.net/en/suite-software-manual/1.8.2/vision-steps/3d-fine-matching-lite.html

    Model:
        name:                               MechVision step name in the project workflow.
        modelSelection:                     Name of the model in the model library
                                            (project/resource/3d_matching/<name>).
        modelFileName:                      Absolute path to the PLY point cloud model file.
        geoCenterFileName:                  Absolute path to the JSON geometric center file.

    Matching method:
        registrationMode:                   Matching algorithm. 0.0 = Surface matching
                                            (objects with recognizable surface features such
                                            as crankshafts or rotors), 1.0 = Edge matching
                                            (flat objects with clear edges such as panels or
                                            brake discs). The model file must match the
                                            selected mode.
        deviationCorrectionCapacity:        Intensity of deviation correction applied to the
                                            matching result. 0.0 = Small, 1.0 = Medium,
                                            2.0 = Large. Excessively large capacity may
                                            reduce accuracy. Default: Small.
        operationApproach:                  Processing speed/accuracy trade-off.
                                            0.0 = High speed, 1.0 = Standard,
                                            2.0 = High accuracy, 3.0 = Extra high accuracy.
                                            Higher accuracy requires longer processing time.
                                            Default: Standard.

    Symmetry:
        rotationStrategy:                   Rotation axis used when the object has rotational
                                            symmetry. 0.0 = X, 1.0 = Y, 2.0 = Z.
        angleStep:                          Angular step size in degrees for symmetry
                                            candidate generation. Range: 0–360.
        minAngle:                           Minimum rotation angle in degrees for symmetry
                                            search range. Default: -180.
        maxAngle:                           Maximum rotation angle in degrees for symmetry
                                            search range. Default: 180.

    Validation:
        onlyConsiderVisibleSurfaceOfModel:  (Surface matching only) When True, only the
                                            visible portion of the model surface is used for
                                            scoring. Recommended for objects where part of
                                            the surface is occluded from the camera (e.g.,
                                            cylinders). Default: False.
        considerErrorofNormalAngles:        When True, incorporates normal angle errors into
                                            the matching score. Default: False.
        scoreLevel:                         Result validation strictness. 0.0 = Low,
                                            1.0 = Standard, 2.0 = High, 3.0 = Ultra-high,
                                            4.0 = Compatibility. Increase when the model
                                            closely resembles the scene point cloud and
                                            false positives occur. Default: Standard.
        confidenceThreshold:               Minimum confidence score for a result to be
                                            accepted. Range: 0–1. Higher values filter out
                                            lower-quality matches. Default: 0.3.

    Output:
        candidateTopNum:                    Maximum number of poses to output per point
                                            cloud. Results are sorted by score; only the
                                            top N highest-scoring poses are returned.
                                            Default: 1.
    """

    name:                               str   = "Fine_Match_Synthetics"
    modelSelection:                     Param = ("", "string", "")
    modelFileName:                      Param = ("", "string", "")
    geoCenterFileName:                  Param = ("", "string", "")

    # Matching method
    registrationMode:                   Param = ("0.0", "double", "")   # 0.0=Surface, 1.0=Edge
    deviationCorrectionCapacity:        Param = ("0.0", "double", "")   # 0.0=Small, 1.0=Medium, 2.0=Large
    operationApproach:                  Param = ("0.0", "double", "")   # 0.0=HighSpeed … 3.0=ExtraHighAccuracy

    # Symmetry
    rotationStrategy:                   Param = ("1.0", "double", "")   # 0.0=X, 1.0=Y, 2.0=Z
    angleStep:                          Param = ("0.0", "double", "")   # degrees, 0–360
    minAngle:                           Param = ("-180.0", "double", "")
    maxAngle:                           Param = ("180.0", "double", "")

    # Validation
    onlyConsiderVisibleSurfaceOfModel:  Param = ("False", "bool", "")
    considerErrorofNormalAngles:        Param = ("False", "bool", "")
    scoreLevel:                         Param = ("0.0", "double", "")   # 0.0=Low … 4.0=Compatibility
    confidenceThreshold:                Param = ("0.0", "double", "")   # 0–1

    # Output
    candidateTopNum:                    Param = ("1", "double", "")