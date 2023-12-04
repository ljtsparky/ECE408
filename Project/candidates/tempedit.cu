temp = __hfma2(mask_4d_rz(feature_out,3,3,0), shared_mem_3d(3, ty+3, tx/2+0), temp);
                temp = __hfma2(mask_4d_rz(feature_out,3,3,1), shared_mem_3d(3, ty+3, tx/2+1), temp);
                temp = __hfma2(mask_4d_rz(feature_out,3,3,2), shared_mem_3d(3, ty+3, tx/2+2), temp);
                temp = __hfma2(mask_4d_rz(feature_out,3,3,3), shared_mem_3d(3, ty+3, tx/2+3), temp);
               