-- Training extract for the same-event pair classifier.
-- One row per labeled candidate pair. The label frame is the inner join: a pair
-- in the feature table with no score stays out of the model. Row order is left
-- as the warehouse returns it so a refit matches the shipped fit; do not add an
-- ORDER BY.
SELECT
  pairs.left_source,
  pairs.left_id,
  pairs.right_source,
  pairs.right_id,
  pairs.emb_score,
  pairs.cross_score,
  pairs.event_score,
  pairs.deadline_score,
  pairs.link_weight,
  pairs.left_tf_idf,
  pairs.right_tf_idf,
  pairs.left_location,
  pairs.right_location,
  pairs.left_doy_deadline_start,
  pairs.left_doy_event_start,
  pairs.left_doy_deadline_duration,
  pairs.left_doy_event_duration,
  pairs.left_event_year,
  pairs.left_deadline_year,
  pairs.right_doy_deadline_start,
  pairs.right_doy_event_start,
  pairs.right_doy_deadline_duration,
  pairs.right_doy_event_duration,
  pairs.right_event_year,
  pairs.right_deadline_year,
  labels.score
FROM staging.candidates_02_pairs_features AS pairs
INNER JOIN staging.pairs_generator_01_same_event AS labels
  ON labels.left_source = pairs.left_source
 AND labels.left_id = pairs.left_id
 AND labels.right_source = pairs.right_source
 AND labels.right_id = pairs.right_id
