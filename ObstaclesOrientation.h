#pragma once

#include "Tools/Streams/AutoStreamable.h"
#include "Tools/Math/Eigen.h"

STREAMABLE(ObstaclesOrientation,
{
  STREAMABLE(Obstacleor,
  {,
    (int) top, /**< The guessed top border of the obstacle in the image. */
    (int) bottom, /**< The lower border of the obstacle in the image. */
    (int) left, /**< The left border of the obstacle in the image. This usually only includes the width at the lower end. */
    (int) right, /**< The right border of the obstacle in the image. This usually only includes the width at the lower end. */
    (bool) bottomFound, /**< Was the lower end of the obstacle found? Otherwise, it was hidden by the lower image border. */
    (bool) fallen, /**< Is the obstacle a player lying on the field? */
    (float)(1.f) probability,
    (float)(-1.f) distance,
    (int) orientation /**< orientation is 1 if front, 0 if backward*/
  });

  /** Draws this percept. */
  void draw() const,

  (std::vector<Obstacleor>) obstacles, /**< All the obstacles found in the current image. */
});