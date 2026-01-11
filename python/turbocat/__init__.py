"""
TurboCat: Next-Generation Gradient Boosting Framework

A high-performance gradient boosting library designed to outperform CatBoost.

Key innovations:
- GradTree: Gradient-based global tree optimization (not greedy splitting)
- Advanced losses: Robust Focal, LDAM, Logit-adjusted, Tsallis
- Tsallis entropy splitting criterion
- Cross-validated target statistics for categoricals
- 3-bit gradient quantization
- SIMD-optimized histogram building (AVX2/AVX-512)
- Optimized batch prediction with cache-friendly processing

Example:
    >>> from turbocat import TurboCatClassifier
    >>> clf = TurboCatClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=-1)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict_proba(X_test)
"""

__version__ = "0.3.0"

import numpy as np


def _convert_to_onnx_classifier(model, initial_types=None):
    """
    Convert TurboCat classifier to ONNX format.

    Note: The current export uses bin indices as split thresholds.
    For accurate inference, input data should be pre-binned using the same
    binning as training data, or use the native TurboCat predict methods.
    """
    try:
        from onnx import helper, TensorProto
        from onnx.helper import make_model, make_graph, make_node, make_tensor_value_info
    except ImportError:
        raise ImportError(
            "ONNX export requires 'onnx' package. Install with: pip install onnx"
        )

    import warnings
    warnings.warn(
        "ONNX export uses histogram bin indices as thresholds. "
        "For accurate predictions, ensure input data is binned identically to training. "
        "Consider using native TurboCat predict() for production inference.",
        UserWarning
    )

    # Get model structure
    trees = model._model.get_booster_dump()
    base_prediction = model._model.get_base_prediction()
    n_features = model._model.get_n_features()
    n_classes = getattr(model, '_params', {}).get('n_classes', 2)

    # Build ONNX TreeEnsembleClassifier attributes
    nodes_treeids = []
    nodes_nodeids = []
    nodes_featureids = []
    nodes_values = []  # thresholds
    nodes_modes = []
    nodes_truenodeids = []
    nodes_falsenodeids = []
    nodes_missing_value_tracks_true = []

    class_treeids = []
    class_nodeids = []
    class_ids = []
    class_weights = []

    for tree_id, tree in enumerate(trees):
        weight = tree['weight']
        nodes = tree['nodes']

        for node_id, node in enumerate(nodes):
            nodes_treeids.append(tree_id)
            nodes_nodeids.append(node_id)

            if node['is_leaf']:
                nodes_featureids.append(0)
                nodes_values.append(0.0)
                nodes_modes.append("LEAF")
                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)
                nodes_missing_value_tracks_true.append(1 if node['default_left'] else 0)

                # Leaf: add class weight
                class_treeids.append(tree_id)
                class_nodeids.append(node_id)
                class_ids.append(1)  # Class 1 for binary
                class_weights.append(float(node['value'] * weight))
            else:
                nodes_featureids.append(node['feature'])
                nodes_values.append(float(node['threshold']))
                nodes_modes.append("BRANCH_LEQ")
                nodes_truenodeids.append(node['left_child'])
                nodes_falsenodeids.append(node['right_child'])
                nodes_missing_value_tracks_true.append(1 if node['default_left'] else 0)

    # Create ONNX graph
    input_name = "input"
    output_label_name = "output_label"
    output_prob_name = "output_probability"

    # Input
    X = make_tensor_value_info(input_name, TensorProto.FLOAT, [None, n_features])

    # Outputs
    output_label = make_tensor_value_info(output_label_name, TensorProto.INT64, [None])
    output_prob = make_tensor_value_info(output_prob_name, TensorProto.FLOAT, [None, 2])

    # TreeEnsembleClassifier node
    tree_node = make_node(
        'TreeEnsembleClassifier',
        inputs=[input_name],
        outputs=[output_label_name, output_prob_name],
        domain='ai.onnx.ml',
        name='TurboCatClassifier',
        nodes_treeids=nodes_treeids,
        nodes_nodeids=nodes_nodeids,
        nodes_featureids=nodes_featureids,
        nodes_values=nodes_values,
        nodes_modes=nodes_modes,
        nodes_truenodeids=nodes_truenodeids,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        class_treeids=class_treeids,
        class_nodeids=class_nodeids,
        class_ids=class_ids,
        class_weights=class_weights,
        classlabels_int64s=[0, 1],
        post_transform='LOGISTIC',
        base_values=[float(base_prediction)],
    )

    graph = make_graph(
        [tree_node],
        'turbocat_classifier',
        [X],
        [output_label, output_prob]
    )

    # Create model with opset for ai.onnx.ml
    from onnx import OperatorSetIdProto
    opset_ml = OperatorSetIdProto()
    opset_ml.domain = 'ai.onnx.ml'
    opset_ml.version = 3

    opset_onnx = OperatorSetIdProto()
    opset_onnx.domain = ''
    opset_onnx.version = 17

    onnx_model = make_model(graph, opset_imports=[opset_onnx, opset_ml])
    onnx_model.ir_version = 8

    return onnx_model


def _convert_to_onnx_regressor(model, initial_types=None):
    """
    Convert TurboCat regressor to ONNX format.

    Note: The current export uses bin indices as split thresholds.
    For accurate inference, input data should be pre-binned using the same
    binning as training data, or use the native TurboCat predict methods.
    """
    try:
        from onnx import helper, TensorProto
        from onnx.helper import make_model, make_graph, make_node, make_tensor_value_info
    except ImportError:
        raise ImportError(
            "ONNX export requires 'onnx' package. Install with: pip install onnx"
        )

    import warnings
    warnings.warn(
        "ONNX export uses histogram bin indices as thresholds. "
        "For accurate predictions, ensure input data is binned identically to training. "
        "Consider using native TurboCat predict() for production inference.",
        UserWarning
    )

    # Get model structure
    trees = model._model.get_booster_dump()
    base_prediction = model._model.base_prediction
    n_features = model._model.get_n_features()

    # Build ONNX TreeEnsembleRegressor attributes
    nodes_treeids = []
    nodes_nodeids = []
    nodes_featureids = []
    nodes_values = []
    nodes_modes = []
    nodes_truenodeids = []
    nodes_falsenodeids = []
    nodes_missing_value_tracks_true = []

    target_treeids = []
    target_nodeids = []
    target_ids = []
    target_weights = []

    for tree_id, tree in enumerate(trees):
        weight = tree['weight']
        nodes = tree['nodes']

        for node_id, node in enumerate(nodes):
            nodes_treeids.append(tree_id)
            nodes_nodeids.append(node_id)

            if node['is_leaf']:
                nodes_featureids.append(0)
                nodes_values.append(0.0)
                nodes_modes.append("LEAF")
                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)
                nodes_missing_value_tracks_true.append(1 if node['default_left'] else 0)

                # Leaf: add target weight
                target_treeids.append(tree_id)
                target_nodeids.append(node_id)
                target_ids.append(0)
                target_weights.append(float(node['value'] * weight))
            else:
                nodes_featureids.append(node['feature'])
                nodes_values.append(float(node['threshold']))
                nodes_modes.append("BRANCH_LEQ")
                nodes_truenodeids.append(node['left_child'])
                nodes_falsenodeids.append(node['right_child'])
                nodes_missing_value_tracks_true.append(1 if node['default_left'] else 0)

    # Create ONNX graph
    input_name = "input"
    output_name = "output"

    X = make_tensor_value_info(input_name, TensorProto.FLOAT, [None, n_features])
    Y = make_tensor_value_info(output_name, TensorProto.FLOAT, [None, 1])

    # TreeEnsembleRegressor node
    tree_node = make_node(
        'TreeEnsembleRegressor',
        inputs=[input_name],
        outputs=[output_name],
        domain='ai.onnx.ml',
        name='TurboCatRegressor',
        nodes_treeids=nodes_treeids,
        nodes_nodeids=nodes_nodeids,
        nodes_featureids=nodes_featureids,
        nodes_values=nodes_values,
        nodes_modes=nodes_modes,
        nodes_truenodeids=nodes_truenodeids,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        target_treeids=target_treeids,
        target_nodeids=target_nodeids,
        target_ids=target_ids,
        target_weights=target_weights,
        n_targets=1,
        post_transform='NONE',
        aggregate_function='SUM',
        base_values=[float(base_prediction)],
    )

    graph = make_graph([tree_node], 'turbocat_regressor', [X], [Y])

    from onnx import OperatorSetIdProto
    opset_ml = OperatorSetIdProto()
    opset_ml.domain = 'ai.onnx.ml'
    opset_ml.version = 3

    opset_onnx = OperatorSetIdProto()
    opset_onnx.domain = ''
    opset_onnx.version = 17

    onnx_model = make_model(graph, opset_imports=[opset_onnx, opset_ml])
    onnx_model.ir_version = 8

    return onnx_model


try:
    from ._turbocat import (
        TurboCatClassifier as _TurboCatClassifier,
        TurboCatRegressor as _TurboCatRegressor,
        print_info,
    )

    # Wrap TurboCatClassifier to return numpy arrays
    class TurboCatClassifier:
        """
        TurboCat Gradient Boosting Classifier.

        A high-performance gradient boosting classifier with advanced features
        including GOSS sampling, multiple loss functions, and SIMD optimization.

        Parameters
        ----------
        n_estimators : int, default=1000
            Number of boosting iterations.
        learning_rate : float, default=0.05
            Step size shrinkage to prevent overfitting.
        max_depth : int, default=6
            Maximum depth of trees.
        max_bins : int, default=255
            Maximum number of bins for histogram building.
        subsample : float, default=0.8
            Fraction of samples used per iteration.
        colsample_bytree : float, default=0.8
            Fraction of features used per tree.
        min_child_weight : float, default=1.0
            Minimum sum of hessians in a leaf.
        lambda_l2 : float, default=1.0
            L2 regularization term.
        loss : str, default='logloss'
            Loss function: 'logloss', 'focal', 'ldam', 'logit_adjusted', 'tsallis'.
        mode : str, default='auto'
            Tree mode: 'small' (regular trees, best quality), 'large' (symmetric trees,
            faster inference), or 'auto' (chooses based on dataset size).
        use_goss : bool, default=True
            Use Gradient-based One-Side Sampling.
        goss_top_rate : float, default=0.2
            Fraction of top gradients to keep in GOSS.
        goss_other_rate : float, default=0.1
            Fraction of other samples to keep in GOSS.
        use_gradtree : bool, default=False
            Use GradTree (gradient-based tree optimization).
        early_stopping_rounds : int, default=50
            Stop if no improvement for this many rounds.
        n_jobs : int, default=-1
            Number of CPU cores to use (-1 = all available). Alias for n_threads.
        n_threads : int, default=-1
            Number of threads (-1 = all available). Same as n_jobs.
        seed : int, default=42
            Random seed for reproducibility.
        verbosity : int, default=1
            Verbosity level (0 = silent, 1 = progress).
        """

        def __init__(
            self,
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            max_bins=255,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            lambda_l2=1.0,
            loss="logloss",
            mode="auto",  # "small", "large", or "auto"
            use_goss=False,
            goss_top_rate=0.2,
            goss_other_rate=0.1,
            use_gradtree=False,
            use_symmetric=False,  # Deprecated: use mode="large" instead
            use_ordered_boosting=False,  # Ordered boosting (like CatBoost)
            early_stopping_rounds=50,
            n_jobs=-1,
            n_threads=None,  # Alias for n_jobs
            seed=42,
            verbosity=1,
        ):
            # Handle n_jobs / n_threads aliasing
            if n_threads is not None:
                effective_threads = n_threads
            else:
                effective_threads = n_jobs

            # Handle deprecated use_symmetric parameter
            if use_symmetric and mode == "auto":
                mode = "large"

            self._model = _TurboCatClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                max_bins=max_bins,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                lambda_l2=lambda_l2,
                loss=loss,
                mode=mode,
                use_goss=use_goss,
                goss_top_rate=goss_top_rate,
                goss_other_rate=goss_other_rate,
                use_gradtree=use_gradtree,
                use_ordered_boosting=use_ordered_boosting,
                early_stopping_rounds=early_stopping_rounds,
                n_threads=effective_threads,
                seed=seed,
                verbosity=verbosity,
            )
            # Store parameters for get_params
            self._params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'max_bins': max_bins,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight,
                'lambda_l2': lambda_l2,
                'loss': loss,
                'mode': mode,
                'use_goss': use_goss,
                'goss_top_rate': goss_top_rate,
                'goss_other_rate': goss_other_rate,
                'use_gradtree': use_gradtree,
                'use_ordered_boosting': use_ordered_boosting,
                'early_stopping_rounds': early_stopping_rounds,
                'n_jobs': n_jobs,
                'seed': seed,
                'verbosity': verbosity,
            }

        def fit(self, X, y, X_val=None, y_val=None, cat_features=None):
            """
            Fit the classifier.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Training data.
            y : array-like of shape (n_samples,)
                Target values.
            X_val : array-like, optional
                Validation data for early stopping.
            y_val : array-like, optional
                Validation targets.
            cat_features : list of int, optional
                Indices of categorical features.

            Returns
            -------
            self : TurboCatClassifier
                Fitted estimator.
            """
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            if X_val is not None:
                X_val = np.asarray(X_val, dtype=np.float32)
                y_val = np.asarray(y_val, dtype=np.float32)
            else:
                X_val = np.array([], dtype=np.float32).reshape(0, X.shape[1] if X.ndim > 1 else 0)
                y_val = np.array([], dtype=np.float32)

            if cat_features is None:
                cat_features = []

            self._model.fit(X, y, X_val, y_val, cat_features)
            return self

        def predict_proba(self, X):
            """
            Predict class probabilities for X.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.

            Returns
            -------
            proba : ndarray of shape (n_samples, n_classes)
                Class probabilities.
            """
            X = np.asarray(X, dtype=np.float32)
            proba = self._model.predict_proba(X)
            return np.asarray(proba, dtype=np.float32)

        def predict_proba_nobinning_fast(self, X):
            """
            FASTEST probability prediction - no binning + cached flat tree data + SIMD.

            This is the recommended method for production inference.
            Uses cached FastFloatEnsemble with column-major transpose for optimal SIMD.
            Only works with symmetric trees (mode='large' or 'auto' with large data).

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.

            Returns
            -------
            proba : ndarray of shape (n_samples, n_classes)
                Class probabilities.
            """
            X = np.asarray(X, dtype=np.float32)
            proba = self._model.predict_proba_nobinning_fast(X)
            return np.asarray(proba, dtype=np.float32)

        def predict(self, X, threshold=0.5):
            """
            Predict class labels for X.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.
            threshold : float, default=0.5
                Decision threshold for binary classification.

            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted class labels.
            """
            proba = self.predict_proba(X)
            if proba.shape[1] == 2:
                # Binary classification
                return (proba[:, 1] >= threshold).astype(np.int32)
            else:
                # Multiclass classification
                return np.argmax(proba, axis=1).astype(np.int32)

        def feature_importance(self):
            """Get feature importance scores."""
            return self._model.feature_importance()

        def save(self, path):
            """Save model to file."""
            self._model.save(path)

        def load(self, path):
            """Load model from file."""
            self._model.load(path)
            return self

        def get_params(self, deep=True):
            """Get parameters (sklearn compatibility)."""
            return self._params.copy()

        def set_params(self, **params):
            """Set parameters (sklearn compatibility)."""
            self._params.update(params)
            return self

        @property
        def n_trees(self):
            """Number of trees in the ensemble."""
            return self._model.n_trees

        def to_onnx(self, initial_types=None):
            """
            Export model to ONNX format.

            Parameters
            ----------
            initial_types : list of tuples, optional
                Input types for ONNX model. If None, uses float32 input.
                Example: [('input', FloatTensorType([None, n_features]))]

            Returns
            -------
            onnx_model : onnx.ModelProto
                ONNX model that can be saved or used with ONNX Runtime.

            Examples
            --------
            >>> clf = TurboCatClassifier()
            >>> clf.fit(X_train, y_train)
            >>> onnx_model = clf.to_onnx()
            >>> import onnx
            >>> onnx.save(onnx_model, 'model.onnx')

            Notes
            -----
            Requires onnx and onnxmltools packages:
            pip install onnx onnxmltools
            """
            return _convert_to_onnx_classifier(self, initial_types)

        def save_onnx(self, path, initial_types=None):
            """
            Save model to ONNX file.

            Parameters
            ----------
            path : str
                Path to save the ONNX model.
            initial_types : list of tuples, optional
                Input types for ONNX model.
            """
            import onnx
            onnx_model = self.to_onnx(initial_types)
            onnx.save(onnx_model, path)

    # Wrap TurboCatRegressor similarly
    class TurboCatRegressor:
        """
        TurboCat Gradient Boosting Regressor.

        Parameters
        ----------
        n_estimators : int, default=1000
            Number of boosting iterations.
        learning_rate : float, default=0.05
            Step size shrinkage.
        max_depth : int, default=6
            Maximum depth of trees.
        loss : str, default='mse'
            Loss function: 'mse', 'mae', 'huber'.
        subsample : float, default=0.8
            Fraction of samples used per iteration.
        colsample_bytree : float, default=0.8
            Fraction of features used per tree.
        use_goss : bool, default=True
            Use Gradient-based One-Side Sampling.
        goss_top_rate : float, default=0.2
            Fraction of top gradients to keep in GOSS.
        goss_other_rate : float, default=0.1
            Fraction of other samples to keep in GOSS.
        early_stopping_rounds : int, default=50
            Stop if no improvement for this many rounds.
        n_jobs : int, default=-1
            Number of CPU cores to use (-1 = all available).
        seed : int, default=42
            Random seed.
        verbosity : int, default=1
            Verbosity level.
        """

        def __init__(
            self,
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            loss="mse",
            mode="auto",  # "small", "large", or "auto"
            subsample=0.8,
            colsample_bytree=0.8,
            use_goss=False,
            goss_top_rate=0.2,
            goss_other_rate=0.1,
            early_stopping_rounds=50,
            lambda_l2=0.0,
            n_jobs=-1,
            n_threads=None,
            seed=42,
            verbosity=1,
        ):
            if n_threads is not None:
                effective_threads = n_threads
            else:
                effective_threads = n_jobs

            self._model = _TurboCatRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                loss=loss,
                mode=mode,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                use_goss=use_goss,
                goss_top_rate=goss_top_rate,
                goss_other_rate=goss_other_rate,
                early_stopping_rounds=early_stopping_rounds,
                lambda_l2=lambda_l2,
                n_threads=effective_threads,
                seed=seed,
                verbosity=verbosity,
            )
            self._params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'loss': loss,
                'mode': mode,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'use_goss': use_goss,
                'lambda_l2': lambda_l2,
                'n_jobs': n_jobs,
                'seed': seed,
                'verbosity': verbosity,
            }

        def fit(self, X, y):
            """Fit the regressor."""
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            if self._params['verbosity'] > 0:
                print(f"[PY-DEBUG] fit() called: X.shape={X.shape}, y.shape={y.shape}")
                print(f"[PY-DEBUG] y stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")

            self._model.fit(X, y)

            if self._params['verbosity'] > 0:
                print(f"[PY-DEBUG] fit() done: n_trees={self._model.n_trees}, base_prediction={self._model.base_prediction:.6f}")

            return self

        def predict(self, X, timing=False):
            """
            Predict target values for X.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.
            timing : bool, default=False
                If True, print timing breakdown for prediction.

            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted values.
            """
            X = np.asarray(X, dtype=np.float32)

            if self._params['verbosity'] > 0:
                print(f"[PY-DEBUG] predict() called: X.shape={X.shape}")
                # Check if X has variance (if all same, predictions will be same)
                x_std = X.std(axis=0)
                zero_var_features = np.sum(x_std < 1e-6)
                print(f"[PY-DEBUG] X stats: mean_std={x_std.mean():.4f}, zero_var_features={zero_var_features}")

            preds = np.asarray(self._model.predict(X, timing), dtype=np.float32)

            if self._params['verbosity'] > 0:
                print(f"[PY-DEBUG] predict() done: preds range=[{preds.min():.6f}, {preds.max():.6f}], unique={len(np.unique(preds))}")
                if len(np.unique(preds)) == 1:
                    print(f"[PY-DEBUG] WARNING: All predictions identical! This is likely a bug in binning or tree traversal.")

            return preds

        def predict_fast(self, X, timing=False):
            """
            Fast prediction that avoids data copy.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.
            timing : bool, default=False
                If True, print timing information.

            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted values.
            """
            X = np.asarray(X, dtype=np.float32)
            return np.asarray(self._model.predict_fast(X, timing), dtype=np.float32)

        def predict_nobinning(self, X, timing=False):
            """
            No-binning prediction using raw float thresholds.

            Fastest path that skips data binning entirely by using the raw
            float thresholds stored in trees for direct comparison.
            Only works with symmetric trees (mode='large' or auto with large data).

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.
            timing : bool, default=False
                If True, print timing information.

            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted values.
            """
            X = np.asarray(X, dtype=np.float32)
            return np.asarray(self._model.predict_nobinning(X, timing), dtype=np.float32)

        def predict_nobinning_fast(self, X, timing=False):
            """
            FASTEST prediction - no binning + cached flat tree data + SIMD.

            This is the recommended method for production inference.
            Uses cached FastFloatEnsemble with column-major transpose for optimal SIMD.
            Only works with symmetric trees (mode='large' or auto with large data).

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Input samples.
            timing : bool, default=False
                If True, print timing information.

            Returns
            -------
            y_pred : ndarray of shape (n_samples,)
                Predicted values.
            """
            X = np.asarray(X, dtype=np.float32)
            return np.asarray(self._model.predict_nobinning_fast(X, timing), dtype=np.float32)

        def debug_info(self):
            """Get debug information about the model."""
            info = self._model.debug_info()
            info['params'] = self._params
            return info

        def tree_info(self):
            """Get information about each tree in the ensemble."""
            return self._model.tree_info()

        def debug_predict(self, X):
            """
            Debug prediction - traces bin values and tree paths.

            Returns detailed information about:
            - Bin variance across samples for each feature
            - Tree traversal path for first few samples
            - Comparison of train vs test data ranges
            """
            X = np.asarray(X, dtype=np.float32)
            return self._model.debug_predict(X)

        @property
        def n_trees(self):
            """Number of trees in the ensemble."""
            return self._model.n_trees

        @property
        def base_prediction_(self):
            """Base prediction (mean for regression)."""
            return self._model.base_prediction

        def get_params(self, deep=True):
            """Get parameters (sklearn compatibility)."""
            return self._params.copy()

        def set_params(self, **params):
            """Set parameters (sklearn compatibility)."""
            self._params.update(params)
            return self

        def save(self, path):
            """Save model to file."""
            self._model.save(path)

        def load(self, path):
            """Load model from file."""
            self._model.load(path)
            return self

        def to_onnx(self, initial_types=None):
            """
            Export model to ONNX format.

            Parameters
            ----------
            initial_types : list of tuples, optional
                Input types for ONNX model. If None, uses float32 input.

            Returns
            -------
            onnx_model : onnx.ModelProto
                ONNX model that can be saved or used with ONNX Runtime.

            Examples
            --------
            >>> reg = TurboCatRegressor()
            >>> reg.fit(X_train, y_train)
            >>> onnx_model = reg.to_onnx()
            >>> import onnx
            >>> onnx.save(onnx_model, 'model.onnx')

            Notes
            -----
            Requires onnx package: pip install onnx
            """
            return _convert_to_onnx_regressor(self, initial_types)

        def save_onnx(self, path, initial_types=None):
            """
            Save model to ONNX file.

            Parameters
            ----------
            path : str
                Path to save the ONNX model.
            initial_types : list of tuples, optional
                Input types for ONNX model.
            """
            import onnx
            onnx_model = self.to_onnx(initial_types)
            onnx.save(onnx_model, path)

except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import C++ extensions: {e}. "
        "Please ensure TurboCat is properly built and installed."
    )
    
    # Provide fallback stubs
    class TurboCatClassifier:
        """Stub - C++ extension not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("TurboCat C++ extension not built")
    
    class TurboCatRegressor:
        """Stub - C++ extension not available"""
        def __init__(self, *args, **kwargs):
            raise ImportError("TurboCat C++ extension not built")
    
    def print_info():
        print("TurboCat (Python fallback - C++ not available)")


__all__ = [
    "TurboCatClassifier",
    "TurboCatRegressor", 
    "print_info",
    "__version__",
]
